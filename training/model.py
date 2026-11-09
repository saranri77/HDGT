"""
model.py  (hdgt_full, see HDGT_FULL_PLAN.md)
========

The full HDGT model, every component behind a flag (train.py's defaults in brackets):

  encoder         HDGT heterogeneous graph network; + the map's drawn shapes as graph nodes the
                  agents attend to in every layer (--use_map_nodes [True], as hdgt_map_gnn)
  goal predictor  (--use_goal_predictor [True]) --goal_predictor [heatmap]: a goal DISTRIBUTION
                  over a grid around each agent's constant-velocity endpoint, in units of its
                  motion scale s_n, from its encoder feature, its track, a crop of the map
                  (--goal_map_crop) and a reachability prior (--goal_reach_prior); the K goals
                  are picked from it (--goal_pick). mtp: hdgt_goal's K WTA heads, no confidence
                  head. Off: no goals anywhere
  diffusion       (--use_diffusion [True]) K goal-conditioned DDIM chains (TransformerDiffusion-
                  Decoder); off: straight lines to the K goals (constant velocity without goals)
  refiner         (--use_refiner [True]) DiffusionRefiner on the K chains
  ranker          (--use_ranker [True]) hdgt_RankNN's ranker: --ranker_models TrajRankNNs with map
                  features and the reference-image patch, scores averaged; orders the K chains,
                  best first. Off: the chains keep the goal order
  map penalties   (--use_map_penalty [False]) goals, denoised paths and refined paths inside shapes
                  their agent may not enter cost their depth (hdgt_map_gnn's penalties; for the
                  goal distribution, the expected depth under it)

The ModeScorer, the TrajScorer and the goal confidence head are gone: the ranker orders the
chains. --bus_as_car (sdd_dataset) sends buses through the car weights, as in hdgt_RankNN.

The main class is `HDGT_model`, so train.py / test_sdd.py (model_module.HDGT_model) work unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import math
import dgl.function as fn
from functools import partial


# =============================================================================
# Map elements: the graph's "map" nodes (--use_map_nodes) and the map geometry
# =============================================================================
NUM_MAP_TYPES = 2          # 0 grass, 1 obstacle
MAP_NODE_FEA_DIM = 3       # [w, h, is_circle], own frame
M2A_EDGE_FEA_DIM = 7       # [rel_c(2), nearest outline point(2), signed dist, cos2d, sin2d]
M2M_EDGE_FEA_DIM = 5       # [rel_c(2), gap, cos2d, sin2d]
# Columns of each agent's shape geometry from the collate fn (map_geom_lis), agent frame
_MG_CX, _MG_CY, _MG_HW, _MG_HH, _MG_COS, _MG_SIN, _MG_CIRC = range(7)


def densify_path(pos):
    """(..., T, 2) positions relative to the agent's last observed position ->
    (..., 2T, 2): each segment's midpoint followed by its end point, the first
    segment starting at the origin. A step of 0.4 s can carry a biker over a
    thin shape, so the midpoints catch paths that jump across one. Same point
    set as sdd_dataset.densify_path_np, which the GT exemption uses.
    """
    prev = torch.cat([torch.zeros_like(pos[..., :1, :]), pos[..., :-1, :]], dim=-2)
    mid = 0.5 * (prev + pos)
    return torch.stack([mid, pos], dim=-2).reshape(*pos.shape[:-2], 2 * pos.shape[-2], 2)


def map_signed_distance(points, geom):
    """Signed distance of points to shapes, negative inside.

    points (N, P, 2), geom (N, S, 7) in the same (agent) frame -> (N, P, S).
    Float32 on purpose: under fp16 autocast the 1e-12 guard underflows and the
    sqrt gradient at 0 turns into inf * 0 = NaN.
    """
    p = points.float().unsqueeze(2)                     # (N, P, 1, 2)
    g = geom.float().unsqueeze(1)                       # (N, 1, S, 7)
    dx = p[..., 0] - g[..., _MG_CX]
    dy = p[..., 1] - g[..., _MG_CY]
    c, s = g[..., _MG_COS], g[..., _MG_SIN]
    u = dx * c + dy * s
    v = -dx * s + dy * c
    qx = u.abs() - g[..., _MG_HW]
    qy = v.abs() - g[..., _MG_HH]
    outside = torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2 + 1e-12) - 1e-6
    inside = torch.maximum(qx, qy).clamp(max=0)
    sd_circ = torch.sqrt(dx * dx + dy * dy + 1e-12) - g[..., _MG_HW]
    return torch.where(g[..., _MG_CIRC] > 0.5, sd_circ, outside + inside)


def map_depth(points, geom, mask):
    """How deep each point is inside the shapes selected by mask, in scaled units
    (x SCALE_FACTOR = pixels).

    points (N, ..., 2), geom (N, S, 7), mask (N, S) bool -> (N, ...). Linear in the
    depth: 0 for a point outside every selected shape, otherwise its distance to
    the nearest way out of the deepest selected shape containing it, so the
    gradient pushes the point out through the closest edge. Overlapping shapes
    are not counted twice.
    """
    n, lead = points.shape[0], points.shape[1:-1]
    flat = points.reshape(n, -1, 2)
    if geom.shape[1] == 0:
        return flat.new_zeros((n,) + tuple(lead)).float()
    depth = F.relu(-map_signed_distance(flat, geom))               # (N, P, S)
    depth = depth * mask.unsqueeze(1).to(depth.dtype)
    return depth.max(dim=-1).values.reshape((n,) + tuple(lead))


# =============================================================================
# Basic Building Blocks (unchanged from original)
# =============================================================================

class BN1D(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_in)
    def forward(self, x):
        if len(x.shape) == 3:
            return self.bn(x.transpose(1, 2)).transpose(1, 2)
        if len(x.shape) == 2:
            is_single_input = (x.shape[0] == 1 and self.training == True)
            if is_single_input:
                self.bn.eval()
            res = self.bn(x.unsqueeze(-1)).squeeze(-1)
            if is_single_input:
                self.bn.train()
            return res


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_se1 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.conv_se2 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv_se2(self.act(self.conv_se1(self.avg_pool(x))))) * x


class SEBasicBlock(nn.Module):
    def __init__(self, in_c, out_c, temporal_length, stride=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Conv1d(kernel_size=3, in_channels=in_c, out_channels=out_c, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)
        self.conv2 = nn.Conv1d(kernel_size=3, in_channels=out_c, out_channels=out_c, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = None
        self.se = SEBlock(in_c)
        if stride != 1:
            if temporal_length in [2, 4, 6, 8]:
                self.downsample = nn.Sequential(torch.nn.AvgPool1d(kernel_size=2, stride=2), nn.Conv1d(kernel_size=1, in_channels=in_c, out_channels=out_c, stride=1), nn.BatchNorm1d(out_c))
            else:
                self.downsample = nn.Sequential(torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=1, count_include_pad=False), nn.Conv1d(kernel_size=1, in_channels=in_c, out_channels=out_c, stride=1), nn.BatchNorm1d(out_c))
        elif in_c != out_c:
            self.downsample = nn.Sequential(nn.Conv1d(kernel_size=1, in_channels=in_c, out_channels=out_c, stride=1), nn.BatchNorm1d(out_c))
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        if self.in_c == self.out_c:
            out = self.se(out)
            out = out + identity
        return out


class TemporalBlock(nn.Module):
    def __init__(self, inplanes, temporal_length, args):
        super(TemporalBlock, self).__init__()
        self.layers = [SEBasicBlock(in_c=inplanes, out_c=inplanes, temporal_length=temporal_length, stride=1)]
        self.layers += [SEBasicBlock(in_c=inplanes, out_c=inplanes, temporal_length=temporal_length, stride=2)]
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        out = x
        for _ in range(len(self.layers)):
            out = self.layers[_](out)
        return out


class AgentTemporalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        obs_len = args.obs_len
        kernel_size_dic = {0: obs_len, 1: max(obs_len // 2, 3)}
        self.temporal_layers = torch.nn.ModuleList([TemporalBlock(args.hidden_dim, kernel_size_dic[_], args) for _ in range(2)])
    def forward(self, feat):
        feat = self.temporal_layers[0](feat)
        for _ in range(1, len(self.temporal_layers)):
            feat = self.temporal_layers[_](feat)
        feat = feat[..., -1]
        return feat


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, norm=None, dropout=0.0, prenorm=False):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_out)
        self.act = nn.ReLU(inplace=True)
        self.prenorm = prenorm
        if norm is None:
            self.norm = nn.Identity()
        elif prenorm:
            self.norm = norm(d_in)
        else:
            self.norm = norm(d_hid)
        self.dropout = nn.Identity()
    def forward(self, x):
        if self.prenorm:
            output = self.dropout(self.w_2(self.act(self.w_1(self.norm(x)))))
        else:
            output = self.dropout(self.w_2(self.act(self.norm(self.w_1(x)))))
        return output


class PointNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = MLP(d_in=args.hidden_dim//4, d_hid=args.hidden_dim//4, d_out=args.hidden_dim//8, norm=BN1D)
        self.fc2 = MLP(d_in=args.hidden_dim//8, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//8, norm=BN1D)
        self.fc3 = MLP(d_in=args.hidden_dim//4, d_hid=args.hidden_dim//4, d_out=args.hidden_dim//4, norm=BN1D)
    def forward(self, x):
        out = self.fc1(x)
        out = torch.cat([out, self.fc2(out).max(-2)[0].unsqueeze(-2).repeat(1, x.shape[-2], 1)], dim=-1)
        out = torch.cat([out, self.fc3(out).max(-2)[0].unsqueeze(-2).repeat(1, x.shape[-2], 1)], dim=-1)
        return out


class Agent2embedding(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.act = nn.ReLU(inplace=True)
        self.fea_MLP = MLP(input_dim-2+args.hidden_dim//4, args.hidden_dim//2, args.hidden_dim, norm=BN1D)
    def forward(self, input_dic, shared_coor_encoder):
        x = input_dic["graph_lis"].nodes["agent"].data["a_n_fea"]
        coor_fea = shared_coor_encoder(x[..., :2])
        fea = torch.cat([coor_fea, x[..., 2:]], dim=-1)
        fea = self.fea_MLP(fea)
        return fea.transpose(1, 2)


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, temperature, args):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(args.dropout)
        self.softmax = torch.nn.Softmax(dim=2)
    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        n = float(v.shape[1])
        attn = attn / self.temperature * math.log(n+1, 32)
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout, args):
        super().__init__()
        self.d_in = d_in
        self.out_dim = d_in
        self.norm = nn.LayerNorm(d_in)
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)
        self.w3 = nn.Linear(d_in, d_hid)
        self.act = nn.SiLU()
        if dropout != 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
    def forward(self, x):
        residual = x
        output = self.norm(x)
        output = self.w2(self.act(self.w1(output)) * self.w3(output))
        output = self.dropout(output) + residual
        return output


# =============================================================================
# Heterogeneous GNN: agents, plus the map's shapes as nodes with --use_map_nodes (hdgt_map_gnn)
# =============================================================================

class AgentHetGNN(nn.Module):
    """One layer of agent updates.

    Each agent fuses three streams: itself (self-loop edge), attention over its
    agent neighbours ("other" edges), and with --use_map_nodes, attention over the map
    elements near it ("m2a" edges). Queries are per agent type; keys/values per
    SOURCE type (agent type for "other", map class for "m2a"). Edge states are
    refreshed from [source, edge, destination] like the original HDGT.
    With --use_map_nodes False the module and its parameter names are exactly the
    agent-only version, so old checkpoints still load.
    """
    def __init__(self, args):
        super().__init__()
        self.norm = nn.LayerNorm
        self.hidden_dim = args.hidden_dim
        self.head_dim = args.head_dim
        self.d_model = args.hidden_dim
        self.n_head = self.d_model//args.head_dim
        self.use_map = flag_on(getattr(args, "use_map_nodes", False))
        d_k = self.head_dim
        d_v = self.head_dim
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), args=args)

        self.wqs = nn.ModuleList([nn.Sequential(
            self.norm(self.d_model),
            nn.Linear(self.d_model, self.n_head * d_k, bias=False),
        ) for _ in range(6)])

        self.wkvs = nn.ModuleDict({})
        self.wkvs["other"] = nn.ModuleList([
            nn.Sequential(
                self.norm(self.d_model),
                nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
            ) for _ in range(6)])

        def _attn_fc():
            return nn.Sequential(
                nn.Linear(self.n_head * d_v, self.d_model, bias=True),
                nn.ReLU(inplace=True)
            )
        self.attn_fcs = nn.ModuleList([nn.ModuleDict(
            {"other": _attn_fc(), "m2a": _attn_fc()} if self.use_map else {"other": _attn_fc()}
        ) for agent_type_index in range(6)])

        self.self_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                nn.ReLU(inplace=True)
            ) for _ in range(6)])

        n_streams = 3 if self.use_map else 2   # self, other agents[, map]
        self.out_fc = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim*n_streams, self.hidden_dim, bias=True),
            nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
        ) for _ in range(6)])

        self.out_ffn = nn.ModuleList([PositionwiseFeedForward(args.hidden_dim, args.hidden_dim*4, args.dropout, nn.LayerNorm)
            for _ in range(6)])

        self.etype_dic = {
            "other": (partial(self.message_func, etype="other"), partial(self.reduce_func, etype="other"))
        }
        self.etype2hidden_name = {"other": "a_e_hidden", "m2a": "m2a_e_hidden"}
        self.etype2out_name = {"other": "a_n_hidden_out", "m2a": "m2a_n_hidden_out"}

        self.edge_MLP = {}
        self.edge_MLP["self"] = nn.ModuleList([MLP(args.hidden_dim, args.hidden_dim, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
            for _ in range(6)])
        self.edge_MLP["other"] = nn.ModuleList([MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
            for _ in range(6)])
        if self.use_map:
            # Map attention gets its own query: "which shape matters" is a
            # different question from "which neighbour matters".
            self.wqs_map = nn.ModuleList([nn.Sequential(
                self.norm(self.d_model),
                nn.Linear(self.d_model, self.n_head * d_k, bias=False),
            ) for _ in range(6)])
            self.wkvs["m2a"] = nn.ModuleList([
                nn.Sequential(
                    self.norm(self.d_model),
                    nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
                ) for _ in range(NUM_MAP_TYPES)])
            self.edge_MLP["m2a"] = nn.ModuleList([MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
                for _ in range(NUM_MAP_TYPES)])
        self.edge_MLP = nn.ModuleDict(self.edge_MLP)

    def forward(self, input_dic):
        """Returns (agent node states, [self, other(, m2a)] edge states)."""
        self.a_e_type_dict = input_dic["a_e_type_dict"]
        self.a_n_type_lis = input_dic["a_n_type_lis"]
        with input_dic["graph_lis"].local_scope():
            input_q = input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"]
            self.device = input_q.device
            # Per-type weights write their rows into one shared buffer. The
            # .to(buffer.dtype) on each of these scatters is what makes --amp
            # work: under autocast a Linear returns fp16/bf16 while the buffer
            # was allocated fp32, and an index-assign refuses to mix dtypes.
            # With --amp none it is a no-op.
            q = torch.zeros((input_q.shape[0], self.n_head * self.head_dim), device=self.device)
            for agent_type_index in range(6):
                if len(self.a_n_type_lis[agent_type_index]) != 0:
                    q[self.a_n_type_lis[agent_type_index]] = (self.wqs[agent_type_index](input_q[self.a_n_type_lis[agent_type_index]])).to(q.dtype)

            other_q = q.view(input_q.shape[0], self.n_head, self.head_dim)
            input_dic["graph_lis"].nodes["agent"].data["other_q"] = other_q

            num_nodes = input_dic["graph_lis"].number_of_nodes("agent")
            input_dic["graph_lis"].nodes["agent"].data["a_n_hidden_out"] = torch.zeros(
                (num_nodes, self.n_head * self.head_dim), device=self.device)

            input_dic["graph_lis"].multi_update_all(etype_dict=self.etype_dic, cross_reducer="stack")
            all_out_fea = input_dic["graph_lis"].nodes["agent"].data["a_n_hidden_out"]

            other_out_tmp_in = all_out_fea.squeeze(1) if len(all_out_fea.shape) == 3 else all_out_fea

            if self.use_map:
                g = input_dic["graph_lis"]
                q_map = torch.zeros((num_nodes, self.n_head * self.head_dim), device=self.device)
                for agent_type_index in range(6):
                    if len(self.a_n_type_lis[agent_type_index]) != 0:
                        q_map[self.a_n_type_lis[agent_type_index]] = (self.wqs_map[agent_type_index](input_q[self.a_n_type_lis[agent_type_index]])).to(q_map.dtype)
                g.nodes["agent"].data["m2a_q"] = q_map.view(num_nodes, self.n_head, self.head_dim)
                # Its own pass with its own output name, so the two attention
                # results can never be mixed up by a cross-etype reducer. An
                # agent with no shape in range gets zeros, like an agent with
                # no neighbour does on "other".
                g.nodes["agent"].data["m2a_n_hidden_out"] = torch.zeros(
                    (num_nodes, self.n_head * self.head_dim), device=self.device)
                if g.num_edges("m2a") > 0:
                    g.update_all(partial(self.message_func, etype="m2a"),
                                 partial(self.reduce_func, etype="m2a"), etype="m2a")
                map_out_tmp_in = g.nodes["agent"].data["m2a_n_hidden_out"]

            input_self_fea = input_dic["graph_lis"].edges["self"].data["a_e_hidden"]
            all_out_n_fea = torch.zeros_like(input_q)
            for agent_type_index in range(6):
                if len(self.a_n_type_lis[agent_type_index]) != 0:
                    other_out_fea = self.attn_fcs[agent_type_index]["other"](other_out_tmp_in[self.a_n_type_lis[agent_type_index]])
                    self_fea = self.self_fc[agent_type_index](input_self_fea[self.a_n_type_lis[agent_type_index]])
                    streams = [self_fea, other_out_fea]
                    if self.use_map:
                        streams.append(self.attn_fcs[agent_type_index]["m2a"](map_out_tmp_in[self.a_n_type_lis[agent_type_index]]))
                    out_n_fea = torch.stack(streams, dim=1)
                    out_n_fea = self.out_fc[agent_type_index](out_n_fea.view(out_n_fea.shape[0], -1)) + input_q[self.a_n_type_lis[agent_type_index]]
                    out_n_fea = self.out_ffn[agent_type_index](out_n_fea)
                    all_out_n_fea[self.a_n_type_lis[agent_type_index]] = (out_n_fea).to(all_out_n_fea.dtype)

            out_e_fea_lis = []
            self_e_fea_tmp_in = input_dic["graph_lis"].edges["self"].data["a_e_hidden"]
            other_e_fea_tmp_in = torch.cat([input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][input_dic["graph_lis"].edges(etype="other")[0]], input_dic["graph_lis"].edges["other"].data["a_e_hidden"], input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][input_dic["graph_lis"].edges(etype="other")[1]]], dim=-1)
            
            self_e_fea_tmp_out = torch.zeros_like(self_e_fea_tmp_in)
            other_e_fea_tmp_out = torch.zeros((other_e_fea_tmp_in.shape[0], self.hidden_dim), device=self.device)
            for agent_type_index in range(6):
                if len(self.a_e_type_dict["self"][agent_type_index]) != 0:
                    self_e_fea_tmp_out[self.a_e_type_dict["self"][agent_type_index]] = (self.edge_MLP["self"][agent_type_index](self_e_fea_tmp_in[self.a_e_type_dict["self"][agent_type_index]])).to(self_e_fea_tmp_out.dtype)
                if len(self.a_e_type_dict["other"][agent_type_index]) != 0:
                    other_e_fea_tmp_out[self.a_e_type_dict["other"][agent_type_index]] = (self.edge_MLP["other"][agent_type_index](other_e_fea_tmp_in[self.a_e_type_dict["other"][agent_type_index]])).to(other_e_fea_tmp_out.dtype)
            out_e_fea_lis.append(self_e_fea_tmp_out + input_dic["graph_lis"].edges["self"].data["a_e_hidden"])
            out_e_fea_lis.append(other_e_fea_tmp_out + input_dic["graph_lis"].edges["other"].data["a_e_hidden"])

            if self.use_map:
                # Map -> agent edge refresh from [shape, edge, agent].
                m2a_u, m2a_v = g.edges(etype="m2a")
                m2a_e_hidden = g.edges["m2a"].data["m2a_e_hidden"]
                m2a_e_fea_tmp_in = torch.cat([g.nodes["map"].data["m_n_hidden"][m2a_u], m2a_e_hidden,
                                              g.nodes["agent"].data["a_n_hidden"][m2a_v]], dim=-1)
                m2a_e_fea_tmp_out = torch.zeros((m2a_e_fea_tmp_in.shape[0], self.hidden_dim), device=self.device)
                for map_type_index in range(NUM_MAP_TYPES):
                    idx = self.a_e_type_dict["m2a"][map_type_index]
                    if len(idx) != 0:
                        m2a_e_fea_tmp_out[idx] = (self.edge_MLP["m2a"][map_type_index](m2a_e_fea_tmp_in[idx])).to(m2a_e_fea_tmp_out.dtype)
                out_e_fea_lis.append(m2a_e_fea_tmp_out + m2a_e_hidden)
        return all_out_n_fea, out_e_fea_lis

    def message_func(self, edges, etype):
        # Keys/values are projected per SOURCE type: agent type for "other",
        # map class for "m2a". Indices are edge ids of that etype.
        tmp_input = edges.data[self.etype2hidden_name[etype]]
        tmp_output = torch.zeros((tmp_input.shape[0], self.n_head * self.head_dim * 2), device=self.device)
        type_index_lis = self.a_e_type_dict[etype]
        for src_type_index in range(len(type_index_lis)):
            if len(type_index_lis[src_type_index]) != 0:
                tmp_output[type_index_lis[src_type_index]] = (self.wkvs[etype][src_type_index](tmp_input[type_index_lis[src_type_index]])).to(tmp_output.dtype)
        k, v = tmp_output.view(-1, self.n_head, self.head_dim*2).split(dim=-1, split_size=self.head_dim)
        return {etype+"_k":k, etype+"_v":v}

    def reduce_func(self, nodes, etype):
        out_name = self.etype2out_name[etype]
        if nodes.mailbox[etype+"_k"].shape[1] == 0:
            node_num = nodes.mailbox[etype+"_k"].shape[0]
            return {out_name: torch.zeros((node_num, self.n_head * self.head_dim), device=self.device)}

        node_num, neighbor_num, n_head, hidden = nodes.mailbox[etype+"_k"].shape
        q = nodes.data[etype+"_q"].view(node_num*self.n_head, -1).unsqueeze(1)
        k = nodes.mailbox[etype+"_k"].transpose(1, 2).reshape(node_num*self.n_head, neighbor_num, -1)
        v = nodes.mailbox[etype+"_v"].transpose(1, 2).reshape(node_num*self.n_head, neighbor_num, -1)
        output, attn = self.attention(q, k, v, mask=None)
        return {out_name:output.view(node_num, -1)}


class MapHetGNN(nn.Module):
    """One layer of map-node updates (--use_map_nodes only).

    A shape fuses itself with attention over the shapes near it ("m2m" edges),
    so the pieces of one region (a building drawn as three rectangles, a lawn
    drawn as several circles) share context before agents read them. Map nodes
    never read agents: the map is static scene context, and nothing is
    predicted for it. Weights are per map class (grass / obstacle).

    update_edges=False skips the m2m edge refresh, for the last map layer,
    whose refreshed edges no later layer would read.
    """
    def __init__(self, args, update_edges=True):
        super().__init__()
        self.update_edges = update_edges
        self.hidden_dim = args.hidden_dim
        self.head_dim = args.head_dim
        self.n_head = args.hidden_dim // args.head_dim
        H, d_k = self.hidden_dim, self.head_dim
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), args=args)
        self.wqs = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(H), nn.Linear(H, self.n_head * d_k, bias=False),
        ) for _ in range(NUM_MAP_TYPES)])
        self.wkvs = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(H), nn.Linear(H, self.n_head * d_k * 2, bias=False),
        ) for _ in range(NUM_MAP_TYPES)])
        self.attn_fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.n_head * d_k, H, bias=True), nn.ReLU(inplace=True),
        ) for _ in range(NUM_MAP_TYPES)])
        self.self_fc = nn.ModuleList([nn.Sequential(
            nn.Linear(H, H, bias=True), nn.ReLU(inplace=True),
        ) for _ in range(NUM_MAP_TYPES)])
        self.out_fc = nn.ModuleList([nn.Sequential(
            nn.Linear(H * 2, H, bias=True),
            nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity(),
        ) for _ in range(NUM_MAP_TYPES)])
        self.out_ffn = nn.ModuleList([PositionwiseFeedForward(H, H * 4, args.dropout, nn.LayerNorm)
                                      for _ in range(NUM_MAP_TYPES)])
        if self.update_edges:
            self.edge_MLP = nn.ModuleList([MLP(H * 3, H * 4, H, nn.LayerNorm, dropout=args.dropout, prenorm=True)
                                           for _ in range(NUM_MAP_TYPES)])

    def forward(self, input_dic):
        """Returns (map node states, m2m edge states — unchanged if not update_edges)."""
        g = input_dic["graph_lis"]
        m_n_type_lis = input_dic["m_n_type_lis"]
        self.m2m_type_lis = input_dic["a_e_type_dict"]["m2m"]
        with g.local_scope():
            h = g.nodes["map"].data["m_n_hidden"]
            self.device = h.device
            num_nodes = h.shape[0]
            q = torch.zeros((num_nodes, self.n_head * self.head_dim), device=self.device)
            for map_type_index in range(NUM_MAP_TYPES):
                idx = m_n_type_lis[map_type_index]
                if len(idx) != 0:
                    q[idx] = (self.wqs[map_type_index](h[idx])).to(q.dtype)
            g.nodes["map"].data["m2m_q"] = q.view(num_nodes, self.n_head, self.head_dim)
            g.nodes["map"].data["m2m_n_hidden_out"] = torch.zeros(
                (num_nodes, self.n_head * self.head_dim), device=self.device)
            if g.num_edges("m2m") > 0:
                g.update_all(self.message_func, self.reduce_func, etype="m2m")
            agg = g.nodes["map"].data["m2m_n_hidden_out"]

            out_n = torch.zeros_like(h)
            for map_type_index in range(NUM_MAP_TYPES):
                idx = m_n_type_lis[map_type_index]
                if len(idx) != 0:
                    fused = torch.cat([self.self_fc[map_type_index](h[idx]),
                                       self.attn_fcs[map_type_index](agg[idx])], dim=-1)
                    o = self.out_fc[map_type_index](fused) + h[idx]
                    out_n[idx] = (self.out_ffn[map_type_index](o)).to(out_n.dtype)

            e_hidden = g.edges["m2m"].data["m2m_e_hidden"]
            if not self.update_edges:
                return out_n, e_hidden
            m2m_u, m2m_v = g.edges(etype="m2m")
            e_in = torch.cat([h[m2m_u], e_hidden, h[m2m_v]], dim=-1)
            e_out = torch.zeros((e_in.shape[0], self.hidden_dim), device=self.device)
            for map_type_index in range(NUM_MAP_TYPES):
                idx = self.m2m_type_lis[map_type_index]
                if len(idx) != 0:
                    e_out[idx] = (self.edge_MLP[map_type_index](e_in[idx])).to(e_out.dtype)
        return out_n, e_out + e_hidden

    def message_func(self, edges):
        tmp_input = edges.data["m2m_e_hidden"]
        tmp_output = torch.zeros((tmp_input.shape[0], self.n_head * self.head_dim * 2), device=self.device)
        for map_type_index in range(NUM_MAP_TYPES):
            idx = self.m2m_type_lis[map_type_index]
            if len(idx) != 0:
                tmp_output[idx] = (self.wkvs[map_type_index](tmp_input[idx])).to(tmp_output.dtype)
        k, v = tmp_output.view(-1, self.n_head, self.head_dim * 2).split(dim=-1, split_size=self.head_dim)
        return {"m2m_k": k, "m2m_v": v}

    def reduce_func(self, nodes):
        k_in = nodes.mailbox["m2m_k"]
        node_num, neighbor_num = k_in.shape[0], k_in.shape[1]
        if neighbor_num == 0:
            return {"m2m_n_hidden_out": torch.zeros((node_num, self.n_head * self.head_dim), device=self.device)}
        q = nodes.data["m2m_q"].view(node_num * self.n_head, -1).unsqueeze(1)
        k = k_in.transpose(1, 2).reshape(node_num * self.n_head, neighbor_num, -1)
        v = nodes.mailbox["m2m_v"].transpose(1, 2).reshape(node_num * self.n_head, neighbor_num, -1)
        output, _ = self.attention(q, k, v, mask=None)
        return {"m2m_n_hidden_out": output.view(node_num, -1)}


class HDGT_encoder(nn.Module):
    """HDGT encoder: agent nodes, plus map nodes with --use_map_nodes."""
    def __init__(self, input_dim, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.use_map = flag_on(getattr(args, "use_map_nodes", False))
        self.shared_coor_encoder = MLP(d_in=2, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//4, norm=None)
        self.shared_rel_encoder = MLP(d_in=4, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//4, norm=None)

        self.agent_emb = Agent2embedding(input_dim, args)
        self.temporal_encoders = torch.nn.ModuleList([AgentTemporalEncoder(args) for _ in range(6)])
        self.agent_e_fea_MLPs = torch.nn.ModuleList([MLP(args.hidden_dim+args.hidden_dim//4, args.hidden_dim*4, args.hidden_dim, norm=BN1D) for _ in range(6)])

        self.num_of_gnn_layer = args.num_of_gnn_layer
        self.agent_gnns = torch.nn.ModuleList([AgentHetGNN(args=args) for _ in range(self.num_of_gnn_layer)])

        if self.use_map:
            H = args.hidden_dim
            # LayerNorm, not the BatchNorm the agent MLPs use: a batch holds many
            # copies of the same few shapes, and a class can be absent or single.
            self.map_emb = nn.ModuleList([MLP(MAP_NODE_FEA_DIM, H//2, H, norm=nn.LayerNorm)
                                          for _ in range(NUM_MAP_TYPES)])
            self.m2a_rel_encoder = MLP(d_in=M2A_EDGE_FEA_DIM, d_hid=H//8, d_out=H//4, norm=None)
            self.m2m_rel_encoder = MLP(d_in=M2M_EDGE_FEA_DIM, d_hid=H//8, d_out=H//4, norm=None)
            self.m2a_e_fea_MLPs = nn.ModuleList([MLP(H + H//4, H*4, H, norm=nn.LayerNorm)
                                                 for _ in range(NUM_MAP_TYPES)])
            self.m2m_e_fea_MLPs = nn.ModuleList([MLP(H + H//4, H*4, H, norm=nn.LayerNorm)
                                                 for _ in range(NUM_MAP_TYPES)])
            # Agent layer i reads the map as it stood after map layer i-1, so a
            # map layer after the second-to-last agent layer would feed nothing:
            # L agent layers need L-1 map layers, and the last of those needs no
            # m2m edge refresh (no later map layer reads the edges).
            n_map_layers = max(self.num_of_gnn_layer - 1, 0)
            self.map_gnns = nn.ModuleList([MapHetGNN(args, update_edges=(i < n_map_layers - 1))
                                           for i in range(n_map_layers)])

    def _init_map_edges(self, g, etype, fea_name, rel_encoder, mlps, type_lis, map_n_fea):
        """Initial state of a map-sourced edge: [relative geometry, source shape]."""
        src, _ = g.edges(etype=etype)
        rel = rel_encoder(g.edges[etype].data[fea_name])
        out = torch.zeros((rel.shape[0], self.hidden_dim), device=rel.device)
        for map_type_index in range(NUM_MAP_TYPES):
            idx = type_lis[map_type_index]
            if len(idx) != 0:
                out[idx] = (mlps[map_type_index](torch.cat([rel[idx], map_n_fea[src[idx]]], dim=-1))).to(out.dtype)
        return out

    def forward(self, input_dic):
        g = input_dic["graph_lis"]
        agent_n_emb = self.agent_emb(input_dic, self.shared_coor_encoder)
        device = agent_n_emb.device
        agent_n_type_indices = [torch.where((g.nodes["agent"].data["a_n_type"]) == _) for _ in range(6)]
        agent_n_fea = torch.zeros((agent_n_emb.shape[0], agent_n_emb.shape[1]), device=device)
        for _ in range(6):
            agent_n_fea[agent_n_type_indices[_]] = (self.temporal_encoders[_](agent_n_emb[agent_n_type_indices[_]])).to(agent_n_fea.dtype)

        agent_edge_types = [('agent', 'self', 'agent'), ('agent', 'other', 'agent')]
        agent_edge_names = ["self", "other"]

        # Per-etype access: with map edges in the graph, g.edata[...] would
        # only list the etypes that happen to carry the feature.
        agent_e_type_lis = torch.cat([g.edges[_].data["a_e_type"] for _ in agent_edge_types], dim=0)
        agent_e_fea_rel_pos = torch.cat([g.edges[_].data["a_e_fea"] for _ in agent_edge_types], dim=0)
        agent_e_type_indices = [torch.where(agent_e_type_lis == _) for _ in range(6)]
        agent_e_src_n_fea = torch.cat([agent_n_fea[g.edges(etype=_)[0],...] for _ in agent_edge_names], dim=0)
        agent_e_fea_rel_pos = self.shared_rel_encoder(agent_e_fea_rel_pos)
        agent_e_fea = torch.zeros((agent_e_fea_rel_pos.shape[0], self.hidden_dim), device=device)
        for _ in range(6):
            agent_e_fea[agent_e_type_indices[_]] = (self.agent_e_fea_MLPs[_](torch.cat([agent_e_fea_rel_pos[agent_e_type_indices[_]], agent_e_src_n_fea[agent_e_type_indices[_]]], dim=-1))).to(agent_e_fea.dtype)
        g.nodes["agent"].data["a_n_hidden"] = agent_n_fea
        agent_e_num_lis_by_etype = np.cumsum([0] + [len(g.edges[_].data["a_e_type"]) for _ in agent_edge_types])
        for _index, _ in enumerate(agent_edge_types):
            g.edges[_].data["a_e_hidden"] = agent_e_fea[agent_e_num_lis_by_etype[_index]:agent_e_num_lis_by_etype[_index+1]]

        if self.use_map:
            m_n_type_lis = input_dic["m_n_type_lis"]
            m_n_fea_in = g.nodes["map"].data["m_n_fea"]
            map_n_fea = torch.zeros((m_n_fea_in.shape[0], self.hidden_dim), device=device)
            for map_type_index in range(NUM_MAP_TYPES):
                idx = m_n_type_lis[map_type_index]
                if len(idx) != 0:
                    map_n_fea[idx] = (self.map_emb[map_type_index](m_n_fea_in[idx])).to(map_n_fea.dtype)
            g.nodes["map"].data["m_n_hidden"] = map_n_fea
            g.edges["m2a"].data["m2a_e_hidden"] = self._init_map_edges(
                g, "m2a", "m2a_e_fea", self.m2a_rel_encoder, self.m2a_e_fea_MLPs,
                input_dic["a_e_type_dict"]["m2a"], map_n_fea)
            g.edges["m2m"].data["m2m_e_hidden"] = self._init_map_edges(
                g, "m2m", "m2m_e_fea", self.m2m_rel_encoder, self.m2m_e_fea_MLPs,
                input_dic["a_e_type_dict"]["m2m"], map_n_fea)

        for i in range(self.num_of_gnn_layer):
            # Both read layer i-1 states, then both are written: a synchronous update.
            output_agent_n_fea, output_in_agent_e_fea = self.agent_gnns[i](input_dic)
            run_map = self.use_map and i < len(self.map_gnns)
            if run_map:
                output_map_n_fea, output_m2m_e_fea = self.map_gnns[i](input_dic)
            g.nodes["agent"].data["a_n_hidden"] = output_agent_n_fea
            g.edges["self"].data["a_e_hidden"] = output_in_agent_e_fea[0]
            g.edges["other"].data["a_e_hidden"] = output_in_agent_e_fea[1]
            if self.use_map:
                g.edges["m2a"].data["m2a_e_hidden"] = output_in_agent_e_fea[2]
            if run_map:
                g.nodes["map"].data["m_n_hidden"] = output_map_n_fea
                g.edges["m2m"].data["m2m_e_hidden"] = output_m2m_e_fea
        return g


# =============================================================================
# Diffusion Components (for diversity)
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, t):
        return self.mlp(t)


class AgentTypeEmbedder(nn.Module):
    """Embeds agent type into vector representations."""
    def __init__(self, num_types, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_types, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, agent_type):
        return self.mlp(self.embedding(agent_type))


class GoalEmbedder(nn.Module):
    """Embeds goal position into vector representations."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Goal is 2D position
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, goal_pos):
        """
        Args:
            goal_pos: (batch, 2) tensor of goal positions
        Returns:
            (batch, hidden_dim) tensor of goal embeddings
        """
        return self.mlp(goal_pos)


class GoalPredictorMTP(nn.Module):
    """
    Multimodal goal predictor (MTP-style): K candidate endpoints.

        agent_features (N, hidden) -> goals (N, K, 2)

    An MLP trunk feeds a linear head. Multimodality is enforced by the
    winner-takes-all loss (computed in HDGT_model.forward), which lets each of
    the K heads specialize on a distinct future mode instead of averaging them.
    No latent, no KL — nothing can collapse.

    There is no confidence head: ordering the K modes is the ranker's job, done
    on the finished chains. This is hdgt_goal's predictor as it is built when a
    scorer ranks the chains (with_conf_head=False there), so the trunk receives
    gradient ONLY from the WTA regression. That matters: in scaled units the
    WTA MSE is ~1e-2 while a 20-way cross-entropy is ~2-3, so a conf head on
    this shared trunk trains the trunk almost entirely for ranking and starves
    the regression it is sharing.
    """
    def __init__(self, hidden_dim, num_modes):
        super().__init__()
        self.num_modes = num_modes
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.goal_head = nn.Linear(hidden_dim, num_modes * 2)

    def forward(self, agent_features):
        h = self.trunk(agent_features)
        goals = self.goal_head(h).view(-1, self.num_modes, 2)  # (N, K, 2)
        return goals


# =============================================================================
# Goal distribution (hdgt_goal_heatmap, GOAL_HEATMAP_PLAN.md)
#
# Per target agent n (scaled units, agent frame: relative to the last observation, +x along the
# heading there): the motion scale s_n = max(kinematic scale, class floor) and the
# constant-velocity endpoint cv_end_n = pred_len x the mean velocity of the last 3 observed steps.
# The goal grid is G x G cells r in [-R, R]^2 (--goal_grid_cells, --goal_grid_range); cell r is the
# goal cv_end_n + s_n * r. GoalPredictorHeatmap gives every agent a distribution over the cells
# (cross-entropy against a Gaussian around the true goal), from its encoder feature and observed
# track, a crop of the map around it and a reachability prior; pick_goals takes the K goals from
# that distribution, most likely first.
# =============================================================================

GOAL_SCALE_MOMENTUM = 0.99    # EMA momentum of each class's mean GT displacement S_c
GOAL_SCALE_WARMUP = 50        # agents a class must have seen before its floor replaces --goal_scale_min
GOAL_OFF_FACTOR = 1.5         # GOAL HEAD report: a goal is "off" beyond 1.5 * max(s_n, ||y_n||)
GOAL_CLOSE_RHO = 0.15         # GOAL HEAD report: two goals nearer than this x s_n count as "close"
MAP_CLASS_GRASS, MAP_CLASS_OBSTACLE = 0, 1   # the collate's map_cls_lis (padding is -1)
CROP_AHEAD = (-0.75, 2.25)    # map crop along the heading, s_n units (columns)
CROP_SIDE = 1.5               # map crop sideways, +-, s_n units (rows)
REACH_BLOCK = 8.0             # travel distance: entering a forbidden cell costs this many steps
REACH_ITERS = 120             # relaxation sweeps (the grid diagonal is ~85 steps)
REACH_CAP = 6.0               # travel distances are capped here, s_n units
NEIGHBOURS_8 = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5))
GOAL_PICK_METHODS = ("kmeans", "flat", "reserve", "sinkhorn")


def kinematic_scale(obs_pos, pred_len):
    """How far each agent plausibly coasts in pred_len steps, before the class floor (scaled):
    max(3-step mean speed, last-step speed) * pred_len. The 3-step speed divides box-centre
    jitter by 3; the last step keeps real accelerations. obs_pos (N, obs_len, 2) -> (N,)"""
    lag = min(3, obs_pos.shape[1] - 1)
    v_short = (obs_pos[:, -1] - obs_pos[:, -2]).norm(dim=-1)
    v_long = (obs_pos[:, -1] - obs_pos[:, -1 - lag]).norm(dim=-1) / lag
    return torch.max(v_long, v_short) * pred_len


def motion_scale(obs_pos, floor, pred_len):
    """s_n = max(kinematic scale, the agent's class floor), scaled; floor (N,)"""
    return torch.max(kinematic_scale(obs_pos, pred_len), floor)


def class_floors(class_scale, class_count, scale_min, class_frac, warmup=GOAL_SCALE_WARMUP):
    """s_min(c) = max(--goal_scale_min, --goal_scale_class_frac * S_c) for a class that has seen
    `warmup` training agents, --goal_scale_min before that: (6,) in, (6,) out, scaled. A biker or
    car stopped at an intersection keeps a biker- or car-sized plausible region"""
    floor = torch.clamp(class_frac * class_scale, min=scale_min)
    return torch.where(class_count >= warmup, floor, torch.full_like(floor, scale_min))


class GoalClassScale(nn.Module):
    """S_c: running mean (EMA, momentum 0.99; a class's first batch sets it) of the mean GT
    displacement ||y_n|| of each class seen in training, scaled, with the number of agents behind
    it -- the class prior of the motion-scale floor (see class_floors), never on the inference
    path. Non-persistent buffers, updated in place; train.py saves them in the checkpoint's extra
    dict ("goal_class_scale", "goal_class_count") for test_sdd.py and for a resumed run"""
    def __init__(self, num_classes=6, momentum=GOAL_SCALE_MOMENTUM):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("scale", torch.zeros(num_classes), persistent=False)
        self.register_buffer("count", torch.zeros(num_classes, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def update(self, gt_goal, cls):
        disp = gt_goal.float().norm(dim=-1)
        n_c = torch.bincount(cls, minlength=self.scale.numel())
        mean_c = torch.zeros_like(self.scale).index_add_(0, cls, disp) / n_c.clamp(min=1)
        ema = torch.where(self.count > 0, self.momentum * self.scale + (1.0 - self.momentum) * mean_c, mean_c)
        self.scale.copy_(torch.where(n_c > 0, ema, self.scale))
        self.count.add_(n_c)

    def load(self, scale, count=None):
        """Restore from a checkpoint's extra dict (lists of 6). Without counts, every class with a
        scale counts as warmed up"""
        scale = torch.as_tensor(scale, dtype=self.scale.dtype)
        if count is None:
            count = torch.where(scale > 0, torch.full_like(scale, GOAL_SCALE_WARMUP), torch.zeros_like(scale))
        self.scale.copy_(scale)
        self.count.copy_(torch.as_tensor(count).long())


def cv_endpoint(obs_pos, pred_len):
    """Constant-velocity endpoint (N, 2), scaled, agent frame: pred_len x the mean velocity of the
    last 3 observed steps (the 3-step speed of kinematic_scale, with its direction)"""
    lag = min(3, obs_pos.shape[1] - 1)
    return (obs_pos[:, -1] - obs_pos[:, -1 - lag]) / lag * pred_len


def goal_grid(cells, extent):
    """(cells * cells, 2) cell centres r in [-extent, extent]^2; the first coordinate (along the
    heading) is the slow index"""
    axis = torch.linspace(-extent, extent, cells)
    return torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)


def map_inside(points, geom, select, chunk=16):
    """(N, P) bool: the point is inside at least one selected shape. points (N, P, 2), geom
    (N, S, 7) and select (N, S) bool, in each agent's frame (the collate's map geometry; see
    map_signed_distance). In chunks of agents, since (N, P, S) temporaries can be large"""
    n, p = points.shape[:2]
    out = torch.zeros((n, p), dtype=torch.bool, device=points.device)
    if geom.shape[1] == 0:
        return out
    for i in range(0, n, chunk):
        sel = select[i:i + chunk]
        if sel.any():
            inside = map_signed_distance(points[i:i + chunk], geom[i:i + chunk]) <= 0      # (n_i, P, S)
            out[i:i + chunk] = (inside & sel.unsqueeze(1)).any(dim=-1)
    return out


def map_crop(geom, cls, forbid, img_geom, s_n, cells):
    """The map around each agent, (N, 4, cells, cells), on a grid in its own frame scaled by s_n:
    columns CROP_AHEAD along the heading, rows +-CROP_SIDE sideways. Channels: grass, obstacle,
    forbidden for this agent (obstacles, and grass for all but pedestrians), inside the image"""
    n, dev = s_n.shape[0], s_n.device
    rows, cols = torch.meshgrid(torch.linspace(-CROP_SIDE, CROP_SIDE, cells, device=dev),
                                torch.linspace(CROP_AHEAD[0], CROP_AHEAD[1], cells, device=dev), indexing="ij")
    pts = torch.stack([cols, rows], dim=-1).reshape(1, -1, 2) * s_n.view(-1, 1, 1)    # (N, C*C, 2)
    every = torch.ones((n, 1), dtype=torch.bool, device=dev)
    crop = torch.stack([map_inside(pts, geom, cls == MAP_CLASS_GRASS),
                        map_inside(pts, geom, cls == MAP_CLASS_OBSTACLE),
                        map_inside(pts, geom, forbid),
                        map_inside(pts, img_geom.unsqueeze(1), every)], dim=1)
    return crop.float().view(n, 4, cells, cells)


def grid_allowed(geom, forbid, img_geom, cv_end, s_n, grid):
    """(N, G*G) bool: the goal cell is inside the image and outside every shape forbidden for
    the agent"""
    pts = cv_end.unsqueeze(1) + s_n.view(-1, 1, 1) * grid.unsqueeze(0)            # (N, G*G, 2)
    every = torch.ones((s_n.shape[0], 1), dtype=torch.bool, device=s_n.device)
    return map_inside(pts, img_geom.unsqueeze(1), every) & ~map_inside(pts, geom, forbid)


def travel_distance(allowed, start, step):
    """(N, G, G) shortest travel distance, s_n units, from each agent's cell start (N, 2) to every
    cell of its goal grid, 8-connected; entering a forbidden cell costs REACH_BLOCK steps (a soft
    wall, so an agent standing on one is not trapped). Capped at REACH_CAP"""
    n, g = allowed.shape[0], allowed.shape[1]
    big = 1e4
    cost = torch.where(allowed, torch.ones_like(allowed, dtype=torch.float32),
                       torch.full_like(allowed, REACH_BLOCK, dtype=torch.float32))
    dist = torch.full((n, g, g), big, device=allowed.device)
    dist[torch.arange(n, device=allowed.device), start[:, 0], start[:, 1]] = 0.0
    for sweep in range(REACH_ITERS):
        padded = F.pad(dist, (1, 1, 1, 1), value=big)
        best = dist
        for di, dj, w in NEIGHBOURS_8:
            best = torch.minimum(best, padded[:, 1 + di:1 + di + g, 1 + dj:1 + dj + g] + w * cost)
        if sweep % 20 == 19 and torch.equal(best, dist):
            break
        dist = best
    return (dist * step).clamp(max=REACH_CAP)


def reach_features(allowed, cv_end, s_n, grid, cells, extent):
    """(N, G*G, 4) per goal cell: travel distance from the agent, straight distance, detour, allowed"""
    n = allowed.shape[0]
    step = 2.0 * extent / (cells - 1)
    r0 = -cv_end / s_n.unsqueeze(-1)                       # the agent's own position, grid units
    start = ((r0 + extent) / step).round().long().clamp(0, cells - 1)
    travel = travel_distance(allowed.view(n, cells, cells), start, step).view(n, -1)
    straight = (grid.unsqueeze(0) - r0.unsqueeze(1)).norm(dim=-1)
    return torch.stack([travel, straight, travel - straight, allowed.float()], dim=-1)


def goal_soft_target(r, grid, sigma, extent):
    """(N, G*G) training target: a Gaussian (std sigma, grid units) around the true goal r (N, 2),
    clamped onto the grid, normalised over the cells"""
    d2 = ((grid.unsqueeze(0) - r.clamp(-extent, extent).unsqueeze(1)) ** 2).sum(dim=-1)
    return torch.softmax(-d2 / (2.0 * sigma ** 2), dim=1)


def _kmeans_cells(prob, grid, k, iters, init=None):
    """Weighted k-means of the cells (weights prob (B, C)), deterministic: the most likely cell
    first, then each time the cell with the largest prob x squared distance to those chosen.
    (B, k, 2)"""
    b = prob.shape[0]
    cells = grid.unsqueeze(0).expand(b, -1, -1)
    cent = grid[prob.argmax(dim=1)].unsqueeze(1) if init is None else init
    while cent.shape[1] < k:
        w = (prob + 1e-12) * torch.cdist(cells, cent).min(dim=-1).values ** 2
        cent = torch.cat([cent, grid[w.argmax(dim=1)].unsqueeze(1)], dim=1)
    for _ in range(iters):
        own = F.one_hot(torch.cdist(cells, cent).argmin(dim=-1), cent.shape[1]).to(prob.dtype) * prob.unsqueeze(-1)
        mass = own.sum(dim=1)
        new = torch.einsum("bck,cd->bkd", own, grid) / mass.unsqueeze(-1).clamp_min(1e-12)
        cent = torch.where(mass.unsqueeze(-1) > 1e-9, new, cent)
    return cent


def _sinkhorn_cells(prob, grid, k, floor, eps, iters, top_cells, outer, kmeans_iters):
    """Balanced optimal-transport quantisation: each goal must carry at least floor / k of the
    agent's probability (floor 0 = k-means, 1 = equal mass per goal). Sinkhorn on the top_cells
    most likely cells, k-means start. (B, k, 2)"""
    b = prob.shape[0]
    pv, pi = prob.topk(min(top_cells, prob.shape[1]), dim=1)
    pv = pv / pv.sum(dim=1, keepdim=True)
    xs = grid[pi]                                                        # (B, C, 2)
    cent = _kmeans_cells(prob, grid, k, kmeans_iters)
    for _ in range(outer):
        cost = torch.cdist(xs, cent) ** 2                                # (B, C, k)
        usage = torch.zeros((b, k), device=prob.device).scatter_add_(1, cost.argmin(dim=-1), pv)
        target = (1.0 - floor) * usage + floor / k
        log_k = -cost / eps
        log_p, log_q = pv.clamp_min(1e-12).log(), (target / target.sum(dim=1, keepdim=True)).log()
        log_v = torch.zeros((b, k), device=prob.device)
        for _ in range(iters):
            log_u = log_p - torch.logsumexp(log_k + log_v.unsqueeze(1), dim=2)
            log_v = log_q - torch.logsumexp(log_k + log_u.unsqueeze(2), dim=1)
        plan = (log_k + log_u.unsqueeze(2) + log_v.unsqueeze(1)).exp()
        cent = torch.einsum("bck,bcd->bkd", plan, xs) / plan.sum(dim=1).unsqueeze(-1).clamp_min(1e-12)
    return cent


@torch.no_grad()
def pick_goals(prob, grid, k, method="kmeans", iters=25, alpha=0.5, reserve=4, reserve_alpha=0.25,
               floor=0.5, eps=0.002, sinkhorn_iters=40, top_cells=1000, outer=15):
    """k goals per agent from its distribution prob (N, G*G) over grid (G*G, 2). Returns the goal
    cells r (N, k, 2), most likely first, and each goal's probability (N, k): the mass of the cells
    nearest to it. method (--goal_pick):
      kmeans   : weighted k-means on prob (the best mean error)
      flat     : k-means on prob^alpha, normalised (< 1 spreads the goals: fewer misses)
      reserve  : k - reserve by k-means, then reserve more, each the cell with the largest
                 prob^reserve_alpha x squared distance to the goals so far (rare futures)
      sinkhorn : balanced optimal-transport quantisation, each goal >= floor / k of the mass"""
    prob = prob.float()
    if method == "kmeans":
        cent = _kmeans_cells(prob, grid, k, iters)
    elif method == "flat":
        q = prob ** alpha
        cent = _kmeans_cells(q / q.sum(dim=1, keepdim=True), grid, k, iters)
    elif method == "reserve":
        cent = _kmeans_cells(prob, grid, k - reserve, iters)
        q = prob ** reserve_alpha
        q = q / q.sum(dim=1, keepdim=True)
        cells = grid.unsqueeze(0).expand(prob.shape[0], -1, -1)
        while cent.shape[1] < k:
            w = (q + 1e-12) * torch.cdist(cells, cent).min(dim=-1).values ** 2
            cent = torch.cat([cent, grid[w.argmax(dim=1)].unsqueeze(1)], dim=1)
    elif method == "sinkhorn":
        cent = _sinkhorn_cells(prob, grid, k, floor, eps, sinkhorn_iters, top_cells, outer, iters)
    else:
        raise ValueError(f"--goal_pick must be one of {GOAL_PICK_METHODS}, got {method!r}")
    own = torch.cdist(grid.unsqueeze(0).expand(prob.shape[0], -1, -1), cent).argmin(dim=-1)
    mass = torch.zeros((prob.shape[0], k), device=prob.device).scatter_add_(1, own, prob)
    order = mass.argsort(dim=1, descending=True)
    return cent.gather(1, order.unsqueeze(-1).expand(-1, -1, 2)), mass.gather(1, order)


def _conv_out(size, layers=3):
    """Side of the map CNN's output: each stride-2, padding-1, 3x3 convolution halves it, rounded up"""
    for _ in range(layers):
        size = (size + 1) // 2
    return size


class GoalPredictorHeatmap(nn.Module):
    """
    Goal distribution over the goal grid (see the section comment):

        [agent feature, observed track, type]    -> MLP -----------------+
        map crop (4 x C x C, with --goal_map_crop) -> CNN ---------------+-> fuse -> G*G logits
        reachability features (G*G x 4, with --goal_reach_prior) -> per-cell MLP, added to them

    The agent feature arrives detached by default (--goal_detach_encoder), so the goal loss
    trains only this module. No heads and no assignment: every agent trains every cell.
    """
    def __init__(self, agent_dim, obs_dim, cells, hidden=128, use_crop=True, use_reach=True, crop_cells=32):
        super().__init__()
        self.use_crop, self.use_reach = use_crop, use_reach
        d_in = agent_dim + obs_dim + 6
        self.track = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
                                   nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        if use_crop:
            self.cnn = nn.Sequential(
                nn.Conv2d(4, 16, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Flatten(), nn.Linear(32 * _conv_out(crop_cells) ** 2, hidden), nn.ReLU(inplace=True))
            self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(inplace=True))
        self.out = nn.Linear(hidden, cells * cells)
        if use_reach:
            self.reach = nn.Sequential(nn.Linear(4, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))

    def forward(self, agent_feat, obs_feat, type_onehot, crop=None, reach=None):
        h = self.track(torch.cat([agent_feat, obs_feat, type_onehot], dim=-1))
        if self.use_crop:
            h = self.fuse(torch.cat([h, self.cnn(crop)], dim=-1))
        logits = self.out(h)
        if self.use_reach:
            logits = logits + self.reach(reach).squeeze(-1)
        return logits


# The DDIM settling diagnostic records the last DDIM_N_SETTLE steps (a ranker input); the refiner
# takes REFINE_STEPS steps (the ranker's zero effort without it has this length)
DDIM_N_SETTLE = 5
REFINE_STEPS = 5


# =============================================================================
# RankNN — the ranker over the K generated chains
#
# Adapted from the RankNN of "Heteroscedastic Diffusion for Multi-Agent
# Trajectory Modeling" (U2Diff). It stands where hdgt_goal's ModeScorer and
# TrajScorer stood. Built inside HDGT_model and trained ONLINE with the
# generator: every training step it ranks the K refined chains the refiner loss
# was computed on, read detached, by their ADE to the ground truth
# (HDGT_model._rank_chains).
#
# Four differences from the paper, each forced by this setting:
#   - a mode here is one chain for ONE agent, not a whole scene: the generator
#     samples agents independently, so bundling chain k across agents would be
#     inventing a correlation that does not exist;
#   - the temporal encoder is a bi-GRU over T=12, not a Mamba over T=50;
#   - social context enters by cross-attention to neighbour tokens, because the
#     agent axis is the batch axis here rather than an axis inside a mode;
#   - the K axis gets a LEARNED SLOT EMBEDDING, which the paper deliberately
#     omits. Their modes are i.i.d. draws and genuinely exchangeable. Here chain
#     k is bound to goal head k and to modal_emb[k] in DiffusionRefiner, and
#     refiner_loss supervises only the winning chain, so slot quality is
#     systematically uneven. ModeScorer is permutation-equivariant and therefore
#     cannot represent that at all; this can.
#
# The paper's uncertainty channels (bivariate NLL, Cholesky head, Reverse
# Gaussian Sampling, Jacobian propagation) are NOT implemented. Their own
# ablation puts that machinery last by contribution, and two free proxies stand
# in for it: how hard the refiner had to drag a chain, and whether DDIM had
# settled when sampling ended.
# =============================================================================

_RANK_EPS = 1e-6
# A parked car has |v0| ~ 1e-4 in scaled units, and 37% of the test split is
# near-stationary cars. An epsilon in the denominator alone still lets the seam
# ratio reach 1e4 and swamp every other channel after normalisation, so the
# ratio itself is clamped.
_SEAM_RATIO_MAX = 10.0


def _rank_safe_norm(x, dim=-1, keepdim=False):
    return x.norm(dim=dim, keepdim=keepdim).clamp_min(_RANK_EPS)


def _rank_cos(a, b):
    """Cosine between two vector fields, safe and in-domain for acos."""
    return ((a * b).sum(-1) / (_rank_safe_norm(a) * _rank_safe_norm(b))).clamp(-1.0, 1.0)


def build_rank_features(trajs, goals, obs, refine_delta=None, settle=None):
    """Features for TrajRankNN, in SLOT order.

    ONE implementation, shared by training and inference (both through
    HDGT_model._rank_chains). Any divergence between those two paths would be
    a silent bug, so there is deliberately no second copy.

    Args (agent-local scaled frame):
        trajs        (N, K, T, 2) refined candidate POSITIONS, relative to last obs
        goals        (N, K, 2)    candidate goals
        obs          (N, obs_len, 2) observed positions
        refine_delta (N, K, R)    per-refinement-step displacement, or None
        settle       (N, K, S)    per-DDIM-step movement near the end, or None
    Returns:
        state (N, K, T, 8), geom (N, K, 19 + S + 4)
        (the trailing 4 are per-agent kinematics, identical across K)

    The geometry block carries BOTH the features ModeScorer had (path length,
    spread) and the features that can contradict them (seam continuity,
    curvature, deviation from constant velocity). The old scorer could only see
    the first group, so if it learned "longer and curvier is better" it had
    nothing available to argue the other way.
    """
    n, k, t, _ = trajs.shape
    dt, dev = trajs.dtype, trajs.device

    # ---- intermediates ----------------------------------------------------
    v0 = obs[:, -1, :] - obs[:, -2, :]                       # (N,2) last observed velocity
    idx = torch.arange(1, t + 1, device=dev, dtype=dt)
    ballistic_path = v0[:, None, None, :] * idx[None, None, :, None]   # (N,1,T,2)
    consensus_path = trajs.mean(dim=1, keepdim=True)         # (N,1,T,2)
    goal_consensus = goals.mean(dim=1, keepdim=True)         # (N,1,2)

    # first step is measured from the origin: last observed position is 0 here
    prev = torch.cat([torch.zeros_like(trajs[:, :, :1, :]), trajs[:, :, :-1, :]], dim=2)
    step_vel = trajs - prev                                  # (N,K,T,2)

    d_consensus = (trajs - consensus_path).norm(dim=-1)      # (N,K,T)
    d_ballistic = (trajs - ballistic_path).norm(dim=-1)      # (N,K,T)
    cum_len = step_vel.norm(dim=-1).cumsum(dim=-1)           # (N,K,T)

    if refine_delta is None:
        refine_delta = torch.zeros(n, k, 5, device=dev, dtype=dt)
    if settle is None:
        settle = torch.zeros(n, k, 5, device=dev, dtype=dt)
    refine_total = refine_delta.sum(-1)                      # (N,K)

    # ---- per-state channels (8) ------------------------------------------
    # Channel 5 is a per-timestep minimum-Bayes-risk signal: distance to the
    # centre of mass of all K candidates is a strong unsupervised predictor of
    # which candidate is right.
    state = torch.cat([
        trajs,                                        # 1-2 position
        step_vel,                                     # 3-4 step velocity
        d_consensus.unsqueeze(-1),                    # 5   distance to consensus
        d_ballistic.unsqueeze(-1),                    # 6   distance to constant velocity
        cum_len.unsqueeze(-1),                        # 7   path length so far
        refine_total[..., None, None].expand(n, k, t, 1),   # 8 refiner effort
    ], dim=-1)

    # ---- per-mode scalars (19 + S) ---------------------------------------
    end = trajs[:, :, -1, :]
    ballistic_end = (v0 * float(t))[:, None, :]

    seam_cos = _rank_cos(step_vel[:, :, 0, :], v0[:, None, :]).unsqueeze(-1)
    seam_ratio = (_rank_safe_norm(step_vel[:, :, 0, :]) / _rank_safe_norm(v0)[:, None]
                  ).clamp(max=_SEAM_RATIO_MAX).unsqueeze(-1)

    turn = torch.acos(_rank_cos(step_vel[:, :, 1:, :], step_vel[:, :, :-1, :]))

    # ---- per-AGENT kinematic scalars (4), broadcast over K -----------------
    # These replace the 256-d encoder feature the ranker used to receive. That
    # feature was unique per agent-window and let the network memorise "this
    # agent -> that slot" on the train cache (train loss fell, val top-1 ADE
    # rose after epoch 4). What a ranker legitimately needs from the past is
    # coarse and shared by thousands of agents: how fast, how straight, how
    # steady. Four numbers carry that; none of them can single out an agent.
    obs_vel = obs[:, 1:, :] - obs[:, :-1, :]                 # (N, obs_len-1, 2)
    obs_speed = obs_vel.norm(dim=-1)                         # (N, obs_len-1)
    speed0 = obs_speed[:, -1:]                               # last observed speed
    obs_accel = obs_speed[:, -1:] - obs_speed[:, :1]         # speed change over the window
    obs_turn = torch.acos(_rank_cos(obs_vel[:, 1:, :], obs_vel[:, :-1, :])).mean(
        dim=-1, keepdim=True)                                # mean heading change
    obs_straight = ((obs[:, -1, :] - obs[:, 0, :]).norm(dim=-1, keepdim=True)
                    / obs_speed.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    ).clamp(max=1.0)                         # chord / arc, 1 = straight
    kin = torch.cat([speed0, obs_accel, obs_turn, obs_straight], dim=-1)   # (N,4)
    kin = kin[:, None, :].expand(n, k, 4)

    geom = torch.cat([
        goals,                                              # 1-2   intended endpoint
        goals - goal_consensus,                             # 3-4   deviation from consensus
        goals - ballistic_end,                              # 5-6   deviation from const-vel
        end,                                                # 7-8   realised endpoint
        end - goals,                                        # 9-10  goal drift
        (goals - goal_consensus).norm(dim=-1, keepdim=True),  # 11  spread
        cum_len[:, :, -1:],                                 # 12    total path length
        seam_cos,                                           # 13    direction continuity
        seam_ratio,                                         # 14    speed continuity
        turn.mean(-1, keepdim=True),                        # 15    mean curvature
        turn.max(-1).values.unsqueeze(-1),                  # 16    worst kink
        d_ballistic.mean(-1, keepdim=True),                 # 17    whole-path const-vel dev
        d_consensus.mean(-1, keepdim=True),                 # 18    medoid score
        refine_total.unsqueeze(-1),                         # 19    refiner effort
        settle,                                             # 20..  DDIM convergence
        kin,                                                # last 4: agent kinematics
    ], dim=-1)

    return state, geom


# ---------------------------------------------------------------------------
# The ranker's two scene clues (--ranker_map, --ranker_image)
# ---------------------------------------------------------------------------

RANK_N_MAP = 6                  # rank_map_features' channels, appended to the geometry block
RANK_CLEARANCE_CAP_PX = 250.0   # a chain's closest approach to a forbidden shape, clipped
# Share of the image patch behind the agent. sdd_dataset.RANK_PATCH_BEHIND cuts the patches
# with it, and train.py checks that the two agree.
RANK_PATCH_BEHIND = 0.375


def rank_map_features(trajs, obs, geom, forbid, scale):
    """Each chain against the shapes its agent's type may not enter, the measure of
    test_sdd.py's MAP COMPLIANCE. Uses no ground truth.

    trajs (N, K, T, 2) and obs (N, obs_len, 2) in the agent's frame, geom (N, S, 7) the
    shapes in that frame, forbid (N, S) True for the shapes off limits to the agent (padding
    False), scale = pixels per scaled unit -> (N, K, RANK_N_MAP): log(1 + px) of the chain's
    deepest point inside one; the share of its path inside; log(1 + px) of its endpoint's
    depth; its closest approach in units of 50 px, negative inside, clipped at
    RANK_CLEARANCE_CAP_PX; then, per agent, log(1 + px) of how deep its own observed past went
    into one, and log(1 + the number of such shapes).
    """
    n, k = trajs.shape[:2]
    points = densify_path(trajs)                                                  # (N, K, 2T, 2)
    depth = map_depth(points, geom, forbid) * scale                               # (N, K, 2T) px
    if geom.shape[1] == 0:
        signed = trajs.new_full((n, k), float("inf"))
    else:
        signed = map_signed_distance(points.reshape(n, -1, 2), geom)              # (N, K 2T, S)
        signed = signed.masked_fill(~forbid[:, None, :], float("inf")).amin(dim=-1)
        signed = signed.reshape(n, k, -1).amin(dim=-1)                            # (N, K)
    clearance = (signed * scale).clamp(-RANK_CLEARANCE_CAP_PX, RANK_CLEARANCE_CAP_PX) / 50.0
    past = (map_depth(obs, geom, forbid) * scale).amax(dim=-1)                    # (N,)
    count = torch.log1p(forbid.sum(dim=1).float())
    return torch.stack([torch.log1p(depth.amax(dim=-1)), (depth > 0).float().mean(dim=-1),
                        torch.log1p(depth[..., -1]), clearance,
                        torch.log1p(past)[:, None].expand(n, k), count[:, None].expand(n, k)], dim=-1)


class RankSceneEncoder(nn.Module):
    """The video's reference image around the agent -> (N, K, d), added to every chain's
    embedding. It sees where agents can go (walkways, roads, doors) as well as where they
    cannot.

    The patch comes in the agent's frame, cut by sdd_dataset: rows along the agent's y axis,
    columns along its heading, size cells of px image pixels, RANK_PATCH_BEHIND of it behind
    the agent. A small CNN keeps a cell for every 2 x 2 patch cells, its dilated layers
    seeing ~260 px around each; each chain reads the cells under its path, by mean and max
    along it and at its end, plus the share of its path outside the patch; the patch as a
    whole goes to every chain too.

    It learns part of the training videos by heart. Taking away what lets it recognise a
    place (its term less its mean over the chains, no dilation, fewer channels) cost its use
    as much as that, offline; averaging rankers (RankEnsemble) is what removed it. Changing
    each training patch to look like another video of the same scene (turned and zoomed
    about the agent, recoloured, squares blanked) stopped the learning by heart but cost more
    on val: the val and test videos film the same 8 scenes, so part of it carries over.
    """

    def __init__(self, d, size, px, scale, channels=32, dropout=0.2):
        super().__init__()
        self.span, self.scale = size * px, scale
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 1))
        self.chain = nn.Sequential(nn.Linear(3 * channels + 1, d), nn.ReLU(inplace=True),
                                   nn.Dropout(dropout), nn.Linear(d, d))
        self.whole = nn.Linear(channels, d)

    def forward(self, patch, trajs):
        """patch (N, 3, S, S) uint8, trajs (N, K, T, 2) the chains in the agent's frame, scaled"""
        x = patch.float() / 255.0 - 0.5
        if self.training:   # the videos of a scene differ in brightness and contrast
            n = x.shape[0]
            x = (x * (1.0 + 0.3 * (torch.rand(n, 1, 1, 1, device=x.device) - 0.5))
                 + 0.1 * (torch.rand(n, 1, 1, 1, device=x.device) - 0.5))
        fmap = self.cnn(x)                                                        # (N, C, S/2, S/2)
        n, k = trajs.shape[:2]
        points = densify_path(trajs) * self.scale                                 # (N, K, 2T, 2) px
        gx = (points[..., 0] + self.span * RANK_PATCH_BEHIND) / self.span * 2 - 1
        gy = (points[..., 1] + self.span / 2) / self.span * 2 - 1
        along = F.grid_sample(fmap, torch.stack([gx, gy], dim=-1).reshape(n, -1, 1, 2),
                              mode="bilinear", padding_mode="zeros", align_corners=False)
        along = along[..., 0].permute(0, 2, 1).reshape(n, k, -1, fmap.shape[1])   # (N, K, 2T, C)
        outside = ((gx.abs() > 1) | (gy.abs() > 1)).float().mean(dim=-1, keepdim=True)
        per_chain = torch.cat([along.mean(dim=2), along.amax(dim=2), along[:, :, -1], outside], dim=-1)
        return self.chain(per_chain) + self.whole(fmap.mean(dim=(2, 3)))[:, None, :]


class TrajRankNN(nn.Module):
    """Reranks the K chains of one agent. Two outputs, both (N, K):

        err_score -- what you SORT BY, ASCENDING. Low is good. It is an error
                     magnitude, the analogue of the paper's error probability.
                     Sorting it descending inverts the ranker completely and
                     looks exactly like a training failure, so the direction is
                     asserted in one place: HDGT_model.forward.
        log_ade   -- auxiliary absolute magnitude. A within-agent ranking is
                     relative by construction and cannot say whether ALL K
                     candidates are bad; this head can, and it gives every one
                     of the K modes a gradient rather than just the winner.

    Shape of the funnel, following the paper: describe each candidate on its
    own, then compare them against each other.

    With image (dict(size, px, scale): the patch sdd_dataset cuts), a
    RankSceneEncoder adds what lies under each chain in the video's reference
    image, beside the slot and type embeddings.
    """

    def __init__(self, hidden_dim, pred_len, K, n_state=8, n_geom=28,
                 d=64, n_heads=4, n_mode_layers=2, n_types=6, dropout=0.2,
                 norm_momentum=0.01, image=None):
        super().__init__()
        # hidden_dim is kept ONLY for ctx_proj (the neighbour tokens are encoder
        # features of that width). The agent's OWN encoder feature is no longer
        # an input — see the note above feat_proj's former position below.
        #
        # dropout default raised 0.1 -> 0.2 and applied at four more places
        # (inputs, geometry MLP, pooled candidate embedding, head). Measured
        # on the 30-epoch run: train loss fell monotonically 2.07 -> 1.68 while
        # val top-1 ADE got WORSE from epoch 4 on (18.8 -> 21.0 px). The model
        # was fitting the train cache, not the ranking problem. Weight decay is
        # raised to match (1e-4 -> 1e-2, train.py --ranker_weight_decay).
        self.K, self.d = K, d

        # Normalisation buffers. Registered rather than stored loose so they
        # travel with the checkpoint and are provably identical between
        # training and inference. Raw channel scales run from curvature in
        # radians to positions in scaled units, so without this the input layer
        # is dominated by whichever channel happens to be largest. Tracked from
        # every training batch by _track_norm_stats, because the chains being
        # ranked keep changing while the generator trains, and statistics
        # fixed at any one moment would go stale.
        self.register_buffer("state_mean", torch.zeros(n_state))
        self.register_buffer("state_std", torch.ones(n_state))
        self.register_buffer("geom_mean", torch.zeros(n_geom))
        self.register_buffer("geom_std", torch.ones(n_geom))
        self.norm_momentum = norm_momentum
        self.register_buffer("norm_batches", torch.zeros((), dtype=torch.long))

        # Input dropout on the normalised feature channels. Zeroing a random
        # subset of the 8 + 24 channels per forward stops the network from
        # leaning on any single one (spread, path length, one settle step) and
        # forces it to read the candidate from several angles.
        self.in_drop = nn.Dropout(dropout)

        # stage one, part A: the path itself, read in time order. ModeScorer
        # flattened T x 2 into one linear layer, which throws away exactly the
        # structure this encoder exists to capture.
        self.state_proj = nn.Linear(n_state, d)
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)

        # stage one, part B: the per-mode scalars
        self.geom_mlp = nn.Sequential(
            nn.Linear(n_geom, d), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d, d))

        # dropout on the pooled per-candidate embedding, before identity terms
        # and context are added and before the K candidates compare themselves
        self.cand_drop = nn.Dropout(dropout)

        # identity terms: which slot and what kind of agent.
        #
        # REMOVED: feat_proj(agent_feat), the agent's own 256-d encoder feature.
        # It was the memorisation channel. Every agent-window has a unique
        # feature vector, so it works as an ID, and the train cache holds ~19k
        # agents x 3 draws: the cheapest way to lower the loss is "for THIS
        # vector, slot 4 wins", which generalises to nothing. That is exactly
        # the pattern in the training log (train loss down, val top-1 up after
        # epoch 4). The kinematic facts that feature carried and that a ranker
        # legitimately needs -- how fast the agent was going, whether it was
        # turning or accelerating -- now enter as four explicit scalars in
        # build_rank_features (geometry channels 25-28), which are shared by
        # many agents and cannot identify one.
        self.slot_emb = nn.Parameter(torch.randn(K, d) * 0.02)
        self.type_emb = nn.Embedding(n_types, d)

        # social conditioning: each candidate attends to the neighbour tokens
        self.ctx_proj = nn.Linear(hidden_dim, d)
        self.ctx_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                              batch_first=True)
        self.ctx_norm = nn.LayerNorm(d)

        # stage two: compare the K candidates. This is where ranking happens,
        # because "is this candidate good" is only answerable relative to its
        # siblings. Strictly more expressive than ModeScorer's mean-subtraction.
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d, dropout=dropout,
            batch_first=True, norm_first=True)
        self.mode_tf = nn.TransformerEncoder(layer, num_layers=n_mode_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d, 2))

        # the scene around the agent, read along each chain (--ranker_image)
        self.scene_enc = RankSceneEncoder(d, dropout=dropout, **image) if image else None

    @torch.no_grad()
    def _track_norm_stats(self, state, geom):
        """Fold one training batch into the running normalisation statistics.

        A plain average over the batches seen so far, until that would weigh a
        batch less than norm_momentum, and an exponential one from there on.
        The first batches therefore set the scale outright instead of being
        pulled toward the (0, 1) the buffers start at, and later ones follow
        the generator's chains as they change. Each batch's variance is pooled
        with the shift of its mean, so the std is the spread across batches,
        not the smaller spread inside a single one.
        """
        w = (1.0 / (self.norm_batches + 1).float()).clamp_min(self.norm_momentum)
        for x, mean, std in ((state, self.state_mean, self.state_std),
                             (geom, self.geom_mean, self.geom_std)):
            x = x.reshape(-1, x.shape[-1]).float()
            delta = x.mean(dim=0) - mean
            var = ((1 - w) * std ** 2 + w * x.var(dim=0, unbiased=False)
                   + w * (1 - w) * delta ** 2)
            mean.add_(w * delta)
            std.copy_(var.sqrt().clamp_min(1e-3))
        self.norm_batches += 1

    def forward(self, state, geom, agent_types, ctx_tokens=None, ctx_mask=None,
                image=None, positions=None):
        """
        state       (N, K, T, n_state)
        geom        (N, K, n_geom)  -- includes the 4 per-agent kinematic scalars
                    that replaced the encoder feature (see build_rank_features)
        agent_types (N,)  ZERO-BASED, 0..5, matching target_agent_types inside
                    HDGT_model (AGENT_TYPE_NAMES is 1..6). Getting the base
                    wrong collapses pedestrian onto biker and raises nothing.
        ctx_tokens  (N, M, hidden_dim) or None
        ctx_mask    (N, M) bool, True = padding, or None
        image       (N, 3, S, S) uint8, each agent's patch of the reference
                    image, and positions (N, K, T, 2), its chains in its frame,
                    scaled: read only by a ranker built with an image
        """
        n, k, t, _ = state.shape
        assert k <= self.slot_emb.shape[0], (
            f"ranker was trained for K={self.slot_emb.shape[0]}, got K={k}")
        # (x - mean) / std over the training batches. Keeps the absolute scale,
        # which the log_ade head needs to say how wrong a chain is in pixels.
        if self.training:
            self._track_norm_stats(state, geom)
        state = (state - self.state_mean) / self.state_std
        geom = (geom - self.geom_mean) / self.geom_std
        state, geom = self.in_drop(state), self.in_drop(geom)

        # per-candidate temporal encoding; the K axis folds into the batch so
        # every candidate is described independently of its siblings
        h = self.state_proj(state).reshape(n * k, t, self.d)
        h, _ = self.gru(h)
        h = h.mean(dim=1).reshape(n, k, self.d)          # mean-pool over time

        h = self.cand_drop(h + self.geom_mlp(geom))
        h = h + self.slot_emb[None, :k, :]
        # clamp, NOT clamp_: agent_types is already int64 here, so .long()
        # returns the very same tensor and an in-place clamp would silently
        # rewrite the caller's target_agent_types.
        types = agent_types.long().clamp(0, self.type_emb.num_embeddings - 1)
        h = h + self.type_emb(types)[:, None, :]
        # (feat_proj(agent_feat) used to be added here; removed, see __init__)
        if self.scene_enc is not None:
            h = h + self.scene_enc(image, positions)

        if ctx_tokens is not None and ctx_tokens.shape[1] > 0:
            kv = self.ctx_proj(ctx_tokens)
            # An agent whose every context token is padding would make softmax
            # produce NaN, so those rows attend to nothing and keep h unchanged.
            if ctx_mask is not None:
                all_pad = ctx_mask.all(dim=1)
                if all_pad.any():
                    ctx_mask = ctx_mask.clone()
                    ctx_mask[all_pad, 0] = False
            attn, _ = self.ctx_attn(h, kv, kv, key_padding_mask=ctx_mask)
            h = self.ctx_norm(h + attn)

        h = self.mode_tf(h)
        out = self.head(h)                               # (N, K, 2)
        return out[..., 0], out[..., 1]                  # err_score, log_ade


def combine_rank_scores(outputs):
    """[(err_score, log_ade)] of several rankers -> one (err_score, log_ade).

    Each ranker's scores are standardised over the agent's K chains, since only
    their order within an agent counts and each ranker has its own scale, then
    averaged; the log-ADE estimates are averaged. One ranker's output passes
    through unchanged.
    """
    if len(outputs) == 1:
        return outputs[0]
    z = [(e - e.mean(dim=1, keepdim=True)) / e.std(dim=1, keepdim=True).clamp_min(1e-6)
         for e, _ in outputs]
    return torch.stack(z).mean(dim=0), torch.stack([a for _, a in outputs]).mean(dim=0)


def mean_with_nearest(positions, n_near=2):
    """(N, K, T, 2) positions in the ranker's order -> (N, T, 2): each agent's first
    chain averaged with its n_near nearest chains (lowest mean distance over the T
    steps).

    Similar chains follow the same plan and differ by sampling noise, which the mean
    partly cancels. On the lab's val chains this cut bikers' top-1 error by ~1.0 px
    ADE and ~3.0 px FDE, left pedestrians unchanged, and made cars and buses worse,
    so test_sdd.py applies it to bikers only (--biker_top1_mean). Kept apart from
    the K chains: put in place of chain 0 it raised minADE, since the first chain
    is often the best one.
    """
    dist = (positions - positions[:, :1]).norm(dim=-1).mean(dim=-1)   # (N, K); 0 for chain 0
    near = dist.topk(n_near + 1, dim=1, largest=False).indices        # chain 0 and its nearest
    return positions.gather(1, near[:, :, None, None].expand(-1, -1, *positions.shape[2:])).mean(dim=1)


class RankEnsemble(nn.Module):
    """TrajRankNNs from different random starts, trained side by side on the same
    chains, each on its own loss (HDGT_model.forward), and judged as one through
    combine_rank_scores.

    The image encoder learns part of the training videos by heart, a different
    part from each random start, and on new videos those parts point each ranker
    at a different wrong chain; averaging cancels them while the rankers agree
    on what holds everywhere. Offline, five rankers with the image ended 13.52 to
    14.24 px of val regret alone and 13.53 averaged. Without the image they
    already agree, and averaging gains nothing.
    """

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, *args, **kwargs):
        return combine_rank_scores([m(*args, **kwargs) for m in self.members])


def build_ranker(hidden_dim, pred_len, K, n_settle, ranker_map=False, ranker_image=False,
                 image_size=64, image_px=8.0, scale=50.0, n_models=1):
    """The ranker, as HDGT_model and rank_lab.py build it: one TrajRankNN, or
    n_models of them in a RankEnsemble, reading the map features (ranker_map,
    rank_map_features) and the image patch (ranker_image, RankSceneEncoder)."""
    n_geom = 19 + n_settle + 4 + RANK_N_MAP * bool(ranker_map)    # see build_rank_features
    image = dict(size=image_size, px=image_px, scale=scale) if ranker_image else None
    members = [TrajRankNN(hidden_dim, pred_len, K, n_geom=n_geom, image=image)
               for _ in range(n_models)]
    return members[0] if n_models == 1 else RankEnsemble(members)


def rankers_of(ranker):
    """The TrajRankNNs of a ranker built by build_ranker"""
    return list(ranker.members) if isinstance(ranker, RankEnsemble) else [ranker]


# The ranker of a checkpoint trained before the --ranker_* options: one plain TrajRankNN
RANKER_ARG_DEFAULTS = {"ranker_map": False, "ranker_image": False, "ranker_image_size": 64,
                       "ranker_image_px": 8.0, "ranker_models": 1}


def ranker_args_from_checkpoint(checkpoint):
    """The --ranker_* settings a checkpoint was trained with (its saved args), to
    build the model it loads into; RANKER_ARG_DEFAULTS for older checkpoints"""
    saved = checkpoint.get("args", {})
    out = {k: saved.get(k, v) for k, v in RANKER_ARG_DEFAULTS.items()}
    out["ranker_map"], out["ranker_image"] = flag_on(out["ranker_map"]), flag_on(out["ranker_image"])
    out["ranker_image_size"], out["ranker_image_px"] = int(out["ranker_image_size"]), float(out["ranker_image_px"])
    out["ranker_models"] = int(out["ranker_models"])
    return out


# ===========================================================================
# RankNN loss: soft-pick regret + soft-rank Spearman + auxiliary magnitude
#
# All terms are computed per agent over that agent's K candidates, then
# aggregated by type. Each term exists to fix a specific defect of the
# cross-entropy-on-argmin loss that trained ModeScorer.
# ===========================================================================
def soft_rank(x, reg=0.1):
    """Differentiable ranks via pairwise sigmoids: rank_i = sum_j sigmoid((x_i-x_j)/reg).

    O(K^2), which at K=20 is free, and it avoids a torchsort dependency (that
    package is not installed here and needs a compile step).

    x is STANDARDISED per row first, so `reg` is measured in standard deviations
    of that row rather than in whatever unit x happens to carry. Without it the
    same reg means two different things on the two sides of the Spearman loss:
    ade_k is tens of pixels, so ade/0.1 saturates into hard ranks, while
    err_score is O(1) and stays soft. Near-ties are then treated as "within 0.1
    pixels" on one side and "within 0.1 output units" on the other, which is not
    the same statement. After standardising, both mean "within 0.1 of this
    agent's own spread", which is what the loss is trying to express.

    Note this does NOT flatten the gradient w.r.t. x. Because the standardised
    rank is scale-invariant by construction, d(rank)/dx still falls as 1/std, so
    a network whose outputs spread out gets smaller gradients (measured: 1.8e-3
    at std 0.1, 6.1e-6 at std 30). That is the correct behaviour — when outputs
    are far apart, moving one barely changes the ordering — and Adam's
    per-parameter normalisation absorbs most of it.
    """
    x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
    d = x.unsqueeze(-1) - x.unsqueeze(-2)          # (..., K, K) x_i - x_j
    return torch.sigmoid(d / reg).sum(-1)


def spearman_loss(err_score, ade_k, reg=0.1):
    """Negative Spearman correlation between predicted and true ordering.

    Replaces cross-entropy on argmin_k ade_k, which had three problems this
    fixes at once:
      - only the single winner got a gradient; here ALL K modes do;
      - over 20 near-duplicate candidates that argmin is largely decided by DDIM
        seed noise, so it was close to a random label. Ranks move smoothly, so
        near-ties stop mattering;
      - correlation is computed WITHIN one agent, so per-agent difficulty
        cancels out for free. That is also why an absolute log-ADE regression is
        useless for ranking and is kept only as an auxiliary head.

    Both quantities are low-is-good, so a perfect ranker gives correlation +1
    and this loss reaches its minimum of -1.
    """
    ra, rb = soft_rank(err_score, reg), soft_rank(ade_k, reg)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    ra = ra / ra.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rb = rb / rb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return -(ra * rb).sum(-1)                      # (N,) one per agent


def soft_pick_loss(err_score, ade_k, tau=1.0):
    """Expected REGRET of a soft top-1 pick, in pixels.

        p_k  = softmax(-z_k / tau),  z = err_score standardised within the agent
        loss = sum_k p_k * ade_k  -  min_k ade_k

    This is the metric the report reads (top-1 ADE), made differentiable. It
    replaced a truncated Plackett-Luce term, and outranks Spearman, because both
    optimise proxies of it that disagree with it in exactly the cases that
    matter:

      - Spearman rewards getting rank 17-vs-18 right as much as rank 1-vs-2;
        top-1 selection does not care about the former at all.
      - Plackett-Luce's soft target softmax(-ade/tau) is what a good ranker
        should OUTPUT, but the loss is a cross-entropy on it: two candidates
        at 12.0 and 12.1 px are almost interchangeable to the metric yet the
        target still asks for a definite preference, and a candidate at 60 px
        and one at 40 px are penalised the same way once both have target ~0.
        Regret weights each candidate by its actual cost in pixels.

    Standardising err_score before the softmax matters. Without it the network
    can lower the loss by simply scaling its outputs up until p is one-hot,
    which kills the gradient to every other candidate and makes tau meaningless.
    With it, tau is in units of "this agent's own score spread": tau=2 is a
    very soft pick that lets the whole list learn; tau=0.3 is close to the hard
    argmin used at inference. HDGT_model anneals from the first to the second
    over the run.

    Subtracting min ade_k does not change the gradient (it is constant in the
    network's parameters) but makes the value read as pixels of regret, so it
    is directly comparable with the top1_ade - min_ade gap in the report.
    """
    z = (err_score - err_score.mean(-1, keepdim=True)) / \
        err_score.std(-1, keepdim=True).clamp_min(1e-6)
    p = F.softmax(-z / tau, dim=-1)
    return (p * ade_k).sum(-1) - ade_k.min(-1).values          # (N,)


def rank_loss(err_score, log_ade, ade_k, types, lam_sp=0.3, lam_pick=1.0,
              lam_reg=0.2, reg=0.1, tau_pick=1.0, pick_scale=10.0):
    """Full objective, aggregated per type then across types.

    Term weights (defaults changed with the soft-pick loss):
      lam_pick  soft-pick regret (pixels / pick_scale). The primary term now:
                it IS the reported metric, softened.
      lam_sp    Spearman, kept at a small weight. It gives every candidate a
                gradient even when the soft pick has become nearly hard, and
                keeps the full ordering meaningful, which the bridge plan's
                per-candidate confidence needs and top-1 regret does not.
      lam_reg   auxiliary log-ADE regression, unchanged.

    pick_scale converts the regret from pixels (typically 3-30) to O(1) so the
    weights above mean what they say relative to the Spearman term in [-1, 1].

    Picking on (ADE + 0.5 FDE) / 1.5 instead of ADE was tried offline: neither
    the pick's ADE nor its FDE moved, since a chain good on one is good on both.

    Type-level aggregation is not cosmetic: pedestrians outnumber bikers by an
    order of magnitude, so a plain mean over the batch tunes the ranker to
    pedestrians alone and the biker rows never move.
    """
    per_agent = (lam_sp * spearman_loss(err_score, ade_k, reg)
                 + lam_pick * soft_pick_loss(err_score, ade_k, tau_pick) / pick_scale
                 + lam_reg * F.smooth_l1_loss(
                     log_ade, torch.log(ade_k + 1e-3), reduction="none").mean(-1))
    # sqrt(n) weighting across the types present, not a flat mean over them.
    #
    # A flat mean is what the REPORT does, and that is right for reporting. As a
    # training objective it is not: expected counts in a batch of 256, from the
    # real train split, are ~189 pedestrians, 40 bikers, 23 cars and one or two
    # each of bus, cart and skater. Weighting the six equally sends about half
    # the gradient to the ~2% of agents that are bus/cart/skater, and lets a
    # single skater count for as much as 189 pedestrians. Those three types are
    # too rare in SDD to estimate anything reliably from.
    #
    # sqrt(n) sits between that and plain agent-averaging (which would hand
    # pedestrians ~74% and bikers 15%): roughly 47/22/17% for pedestrian, biker
    # and car with ~14% shared by the rare three. Rare types still move the
    # ranker; one sample no longer swings a step.
    ts = types.unique()
    terms = torch.stack([per_agent[types == t].mean() for t in ts])
    w = torch.stack([(types == t).sum() for t in ts]).to(terms.dtype).sqrt()
    return (terms * w).sum() / w.sum()


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.gamma_proj = nn.Linear(hidden_dim, hidden_dim)
        self.beta_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cond):
        normalized = self.norm(x)
        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)
        
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        return normalized * (1 + gamma) + beta


class TransformerDenoiserBlock(nn.Module):
    """Single transformer block with AdaLN for denoising."""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.adaln_self = AdaLayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.adaln_cross = AdaLayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.adaln_ffn = AdaLayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb, context, context_mask=None):
        # Self-attention
        normed = self.adaln_self(x, t_emb)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        # Cross-attention to context
        normed = self.adaln_cross(x, t_emb)
        attn_out, _ = self.cross_attn(normed, context, context, key_padding_mask=context_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward
        normed = self.adaln_ffn(x, t_emb)
        x = x + self.ffn(normed)
        
        return x


class DiffusionSchedule(nn.Module):
    """Cosine noise schedule for diffusion."""
    def __init__(self, num_timesteps=100, beta_start=0.0001, beta_end=0.02, schedule_type='cosine'):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        if schedule_type == 'cosine':
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to x_0 to get x_t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        while sqrt_alpha_cumprod_t.dim() < x_0.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise


class TransformerDiffusionDecoder(nn.Module):
    """Transformer-based diffusion decoder for trajectory prediction."""
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.pred_len = args.pred_len
        self.num_samples = args.num_prediction
        self.num_timesteps = getattr(args, 'diffusion_steps', 100)
        
        self.schedule = DiffusionSchedule(
            num_timesteps=self.num_timesteps,
            schedule_type='cosine'
        )
        
        self.traj_embed = nn.Sequential(
            nn.Linear(2, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
        )
        
        # how many trailing DDIM steps the settling diagnostic records
        self.N_SETTLE = DDIM_N_SETTLE
        self.pos_embed = nn.Parameter(torch.randn(1, self.pred_len, self.hidden_dim) * 0.02)
        self.time_embed = TimestepEmbedder(self.hidden_dim)
        self.agent_type_embed = AgentTypeEmbedder(6, self.hidden_dim)
        self.goal_embed = GoalEmbedder(self.hidden_dim)

        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        num_layers = getattr(args, 'diffusion_layers', 6)
        num_heads = self.hidden_dim // args.head_dim
        self.transformer_blocks = nn.ModuleList([
            TransformerDenoiserBlock(self.hidden_dim, num_heads, dropout=args.dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, 2)
        
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def get_context_features(self, agent_ids, input_dic):
        """Gather context features from neighboring agents."""
        device = "cuda:" + str(input_dic["gpu"])
        batch_size = len(agent_ids)
        
        self_features = input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][agent_ids]
        
        context_list = []
        max_neighbors = 0
        
        for agent_id in agent_ids:
            neighbor_eids = input_dic["graph_lis"].in_edges(etype="other", v=agent_id, form="eid")
            if len(neighbor_eids) > 0:
                neighbor_features = input_dic["graph_lis"].edges["other"].data["a_e_hidden"][neighbor_eids]
                context_list.append(neighbor_features)
                max_neighbors = max(max_neighbors, len(neighbor_eids))
            else:
                context_list.append(torch.zeros((0, self.hidden_dim), device=device))
        
        if max_neighbors == 0:
            context = self_features.unsqueeze(1)
            context_mask = None
        else:
            padded_contexts = []
            context_masks = []
            for ctx in context_list:
                if ctx.shape[0] < max_neighbors:
                    padding = torch.zeros((max_neighbors - ctx.shape[0], self.hidden_dim), device=device)
                    padded = torch.cat([ctx, padding], dim=0)
                    mask = torch.cat([
                        torch.zeros(ctx.shape[0], device=device),
                        torch.ones(max_neighbors - ctx.shape[0], device=device)
                    ], dim=0).bool()
                else:
                    padded = ctx
                    mask = torch.zeros(max_neighbors, device=device).bool()
                padded_contexts.append(padded)
                context_masks.append(mask)
            
            neighbor_context = torch.stack(padded_contexts, dim=0)
            context_mask = torch.stack(context_masks, dim=0)
            
            context = torch.cat([self_features.unsqueeze(1), neighbor_context], dim=1)
            context_mask = torch.cat([
                torch.zeros((batch_size, 1), device=device).bool(),
                context_mask
            ], dim=1)
        
        return self.context_proj(context), context_mask

    def forward_denoise(self, noisy_traj, t, agent_features, agent_types, goal_pos, context, context_mask):
        """Forward pass of the denoising network with goal conditioning.

        Noise prediction: predicts the noise (epsilon) added during diffusion,
        while conditioned on goal position for goal-aware generation.
        """
        x = self.traj_embed(noisy_traj)
        x = x + self.pos_embed

        t_emb = self.time_embed(t)
        type_emb = self.agent_type_embed(agent_types)
        goal_emb = self.goal_embed(goal_pos)
        cond_emb = t_emb + type_emb + goal_emb + agent_features

        for block in self.transformer_blocks:
            x = block(x, cond_emb, context, context_mask)

        x = self.final_norm(x)
        predicted_noise = self.output_proj(x)

        return predicted_noise

    def predict_x0_from_noise(self, x_t, t, noise):
        """Recover x0 (clean velocities) from x_t and predicted noise."""
        sqrt_alpha_cumprod_t = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha_cumprod_t.dim() < x_t.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)

        return (x_t - sqrt_one_minus_alpha_cumprod_t * noise) / sqrt_alpha_cumprod_t

    def training_forward(self, agent_features, agent_ids, agent_types, labels, goal_pos, input_dic,
                         goal_target=None, use_goal_penalty=True, map_geom=None, map_mask=None):
        """Training forward pass with noise prediction and alpha_t-weighted goal loss.

        The model predicts noise (epsilon). For the goal penalty, we recover
        x0 (velocities) from the noise prediction, convert to positions via
        cumsum, and compare the final position to the goal. The penalty is
        weighted by alpha_t so it only applies at low noise levels where x0
        recovery is accurate. goal_pos conditions the denoiser; goal_target, when
        given, replaces it as the penalty's target (--goal_cond_noise: the denoiser
        sees a noisy goal, the penalty keeps the clean one). use_goal_penalty False
        (no goal predictor: goal_pos is zero) drops the penalty.

        With map_geom / map_mask (--diffusion_map_penalty), the same recovered
        path is also charged the depth of each of its points inside a shape the
        agent may not enter (densified, so a step cannot jump a thin shape),
        alpha_t-weighted for the same reason. Returns (predicted_noise, noise,
        goal_penalty, map_penalty), map_penalty 0 without map_geom.
        """
        device = agent_features.device
        batch_size = agent_features.shape[0]

        context, context_mask = self.get_context_features(agent_ids, input_dic)

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(labels)
        noisy_traj = self.schedule.q_sample(labels, t, noise)

        # Model predicts noise (epsilon)
        predicted_noise = self.forward_denoise(
            noisy_traj, t, agent_features, agent_types, goal_pos, context, context_mask
        )

        # Goal loss: recover x0 (velocities) from noise, convert to positions,
        # penalize final position vs goal. Weighted by alpha_t for stability.
        predicted_x0 = self.predict_x0_from_noise(noisy_traj, t, predicted_noise)
        predicted_x0 = torch.clamp(predicted_x0, -10, 10)
        predicted_positions = torch.cumsum(predicted_x0, dim=1)  # velocities -> positions
        final_pos = predicted_positions[:, -1, :]  # (batch, 2)
        alpha_t = self.schedule.alphas_cumprod[t].unsqueeze(-1)  # (batch, 1)
        if use_goal_penalty:
            target = goal_pos if goal_target is None else goal_target
            goal_penalty = torch.mean(alpha_t * (final_pos - target) ** 2)
        else:
            goal_penalty = torch.zeros((), device=device)

        if map_geom is not None:
            depth = map_depth(densify_path(predicted_positions), map_geom, map_mask)   # (batch, 2T)
            map_penalty = torch.mean(alpha_t.squeeze(-1) * depth.mean(dim=-1))
        else:
            map_penalty = torch.zeros((), device=device)

        return predicted_noise, noise, goal_penalty, map_penalty

    @torch.no_grad()
    def ddim_sample(self, agent_features, agent_ids, agent_types, goal_pos, input_dic,
                    num_samples=None, ddim_steps=20, collect_settle=False):
        """Fast DDIM sampling with noise prediction and goal conditioning.

        collect_settle: stash how far each chain still moves over the last few
        sampling steps into self.last_settle, (batch, K, N_SETTLE). A chain that
        is still moving when sampling ends never converged, which is a free
        stand-in for the sample-quality signal a variance head would give.
        Slot order, detached, and off by default so normal inference is unchanged.
        """
        if num_samples is None:
            num_samples = self.num_samples

        device = agent_features.device
        batch_size = agent_features.shape[0]

        context, context_mask = self.get_context_features(agent_ids, input_dic)

        agent_features_exp = agent_features.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.hidden_dim)
        agent_types_exp = agent_types.unsqueeze(1).expand(-1, num_samples).reshape(-1)
        # goal_pos can be (batch, 2) at train time or (batch, K, 2) at inference
        if goal_pos.dim() == 3:  # (batch, K, 2) — distinct goal per DDIM chain
            goal_pos_exp = goal_pos.reshape(-1, 2)
        else:                    # (batch, 2) — same goal replicated across K chains
            goal_pos_exp = goal_pos.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 2)
        context_exp = context.unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(-1, context.shape[1], context.shape[2])
        if context_mask is not None:
            context_mask_exp = context_mask.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, context_mask.shape[1])
        else:
            context_mask_exp = None

        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn((batch_size * num_samples, self.pred_len, 2), device=device)

        settle_steps = []
        first_settle_i = len(timesteps) - self.N_SETTLE

        for i, t_idx in enumerate(timesteps):
            x_prev = x if (collect_settle and i >= first_settle_i) else None
            t = torch.full((batch_size * num_samples,), t_idx, device=device, dtype=torch.long)

            # Model predicts noise (epsilon)
            predicted_noise = self.forward_denoise(
                x, t, agent_features_exp, agent_types_exp, goal_pos_exp, context_exp, context_mask_exp
            )

            # Recover x0 from noise prediction
            x_0_pred = self.predict_x0_from_noise(x, t, predicted_noise)
            x_0_pred = torch.clamp(x_0_pred, -10, 10)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t = self.schedule.alphas_cumprod[t_idx]
                alpha_t_prev = self.schedule.alphas_cumprod[t_prev]

                while alpha_t.dim() < x.dim():
                    alpha_t = alpha_t.unsqueeze(-1)
                while alpha_t_prev.dim() < x.dim():
                    alpha_t_prev = alpha_t_prev.unsqueeze(-1)

                # DDIM step: x_{t-1} = √α_{t-1} · x̂₀ + √(1-α_{t-1}) · ε
                x = torch.sqrt(alpha_t_prev) * x_0_pred + torch.sqrt(1 - alpha_t_prev) * predicted_noise
            else:
                x = x_0_pred

            if x_prev is not None:
                # mean over time of the per-step L2 move, one scalar per chain
                settle_steps.append((x - x_prev).norm(dim=-1).mean(dim=-1))

        if collect_settle:
            if settle_steps:
                s = torch.stack(settle_steps, dim=-1)
            else:                      # ddim_steps smaller than N_SETTLE
                s = torch.zeros((batch_size * num_samples, 0), device=device)
            if s.shape[-1] < self.N_SETTLE:   # left-pad so the shape is fixed
                s = F.pad(s, (self.N_SETTLE - s.shape[-1], 0))
            self.last_settle = s.reshape(batch_size, num_samples, self.N_SETTLE).detach()

        trajectories = x.reshape(batch_size, num_samples, self.pred_len, 2)
        return trajectories
    


# =============================================================================
# Refiner (for accuracy) - Following Original HDGT Design
# =============================================================================

class RefineCNN(nn.Module):
    """Temporal CNN block with dilated convolutions and SE attention"""
    def __init__(self, in_c, dilation=1):
        super(RefineCNN, self).__init__()
        self.in_c = in_c
        self.conv1 = nn.Conv1d(in_c, in_c, kernel_size=3, dilation=dilation, padding=dilation)
        self.gn1 = nn.GroupNorm(num_groups=max(1, in_c // 16), num_channels=in_c)
        self.conv2 = nn.Conv1d(in_c, in_c, kernel_size=3, dilation=dilation, padding=dilation)
        self.gn2 = nn.GroupNorm(num_groups=max(1, in_c // 16), num_channels=in_c)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(in_c)
    
    def forward(self, x):
        identity = x
        out = self.conv1(self.act(self.gn1(x)))
        out = self.conv2(self.act(self.gn2(out)))
        out = self.se(out)
        out = out + identity
        out = self.act(out)
        return out


class RefineLayer(nn.Module):
    """Single refinement layer with multi-scale temporal CNNs"""
    def __init__(self, d_in, d_hid, args):
        super().__init__()
        self.obs_len = args.obs_len
        self.in_linear = nn.Linear(d_in, d_hid)
        self.context_mlp = MLP(args.hidden_dim, args.hidden_dim//2, d_hid, norm=nn.LayerNorm, prenorm=True)
        self.fuse_linear = nn.Linear(d_hid * 2, d_hid)
        
        # Multi-scale temporal CNNs (like original HDGT)
        self.cnns = nn.Sequential(
            RefineCNN(d_hid, dilation=1),
            RefineCNN(d_hid, dilation=2),
            RefineCNN(d_hid, dilation=5),
            RefineCNN(d_hid, dilation=1),
        )
        
        self.out_linear = MLP(d_hid, d_hid, 2, norm=None)
    
    def forward(self, x, context):
        """
        Refine trajectory predictions.
        Args:
            x: (batch, num_samples, obs_len+pred_len, 2) - full trajectory
            context: (batch, num_samples, hidden_dim) - context features
        Returns:
            residual: (batch, num_samples, pred_len, 2) - adjustment to predictions
        """
        batch_size, num_samples, total_len, _ = x.shape
        
        # Flatten batch and samples
        x = x.view(batch_size * num_samples, total_len, 2)
        context = context.view(batch_size * num_samples, -1)
        
        # Embed trajectory
        output = self.in_linear(x)
        
        # Inject context at every timestep
        context = self.context_mlp(context).unsqueeze(1).repeat(1, total_len, 1)
        output = self.fuse_linear(torch.cat([context, output], dim=-1))
        
        # Apply temporal CNNs
        output = output.transpose(1, 2)  # (B*K, hidden, time)
        output = self.cnns(output)
        output = output.transpose(1, 2)  # (B*K, time, hidden)
        
        # Extract prediction part (skip observation)
        output = output[:, self.obs_len:, :]
        
        # Predict residual
        residual = self.out_linear(output)
        residual = residual.view(batch_size, num_samples, -1, 2)
        
        return residual


class DiffusionRefiner(nn.Module):
    """
    Refines diffusion-generated trajectories using temporal CNNs.
    Follows original HDGT design with modal embeddings.
    """
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.num_samples = args.num_prediction
        self.refine_num = 5  # 5 refinement iterations (like original HDGT)

        # Modal embeddings (like original HDGT)
        self.modal_emb = nn.Parameter(torch.zeros(self.num_samples, self.hidden_dim // 4))
        nn.init.normal_(self.modal_emb, std=0.02)

        # Goal embedder for goal conditioning
        self.goal_embed = GoalEmbedder(self.hidden_dim)

        # Context projection (now includes goal embedding)
        self.context_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.hidden_dim // 4, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 5 refinement layers
        self.refine_layers = nn.ModuleList([
            RefineLayer(2, self.hidden_dim // 4, args) 
            for _ in range(self.refine_num)
        ])
    
    def forward(self, pred_traj, obs_traj, agent_features, goal_pos, return_all_steps=False):
        """
        Refine diffusion-generated trajectories with goal conditioning.

        Args:
            pred_traj: (batch, num_samples, pred_len, 2) - velocities from diffusion
            obs_traj: (batch, obs_len, 2) - observation positions
            agent_features: (batch, hidden_dim) - encoded agent features
            goal_pos: (batch, 2) - goal positions
            return_all_steps: If True, return all intermediate refinements

        Returns:
            If return_all_steps=True: (refined_traj, goal_penalty) where refined_traj is (batch, refine_num+1, num_samples, pred_len, 2)
            If return_all_steps=False: (refined_traj, goal_penalty) where refined_traj is (batch, num_samples, pred_len, 2)
        """
        batch_size, num_samples, pred_len, _ = pred_traj.shape

        # Convert velocities to positions
        pred_positions = torch.cumsum(pred_traj, dim=2)
        obs_positions = obs_traj
        obs_exp = obs_positions.unsqueeze(1).expand(-1, num_samples, -1, -1)

        # Prepare context (agent features + modal embeddings + goal embedding)
        modal_emb_exp = self.modal_emb.unsqueeze(0).expand(batch_size, -1, -1)
        agent_features_exp = agent_features.unsqueeze(1).expand(-1, num_samples, -1)
        # goal_pos can be (batch, 2) or (batch, K, 2) — GoalEmbedder handles (..., 2)
        if goal_pos.dim() == 3:  # (batch, K, 2) — distinct goal per sample
            goal_emb_exp = self.goal_embed(goal_pos)  # (batch, K, hidden_dim)
        else:                    # (batch, 2) — replicate
            goal_emb_exp = self.goal_embed(goal_pos).unsqueeze(1).expand(-1, num_samples, -1)
        context = torch.cat([agent_features_exp, goal_emb_exp, modal_emb_exp], dim=-1)
        context = self.context_proj(context)
        
        # Start with initial prediction from diffusion
        refined_pred = pred_positions.clone()
        
        # Store all steps: [initial, refine1, refine2, refine3, refine4, refine5]
        all_steps = [refined_pred]
        
        # Iterative refinement (like original HDGT)
        for refine_layer in self.refine_layers:
            # Concatenate observation + current prediction
            # FIXED: Removed .detach() to allow gradient flow between refiner layers
            full_traj = torch.cat([obs_exp, refined_pred], dim=2)

            # Compute residual adjustment
            residual = refine_layer(full_traj, context)

            # Apply residual with small scaling factor for stability
            refined_pred = refined_pred + 0.3 * residual

            # Save this step
            all_steps.append(refined_pred)

        # Compute goal penalty: average distance from final positions to goal across all refinement steps
        if goal_pos.dim() == 3:  # (batch, K, 2) — align with (batch, steps, K, 2)
            goal_pos_exp = goal_pos.unsqueeze(1).expand(-1, len(all_steps), -1, -1)
        else:                    # (batch, 2)
            goal_pos_exp = goal_pos.unsqueeze(1).unsqueeze(1).expand(-1, len(all_steps), num_samples, -1)
        all_final_positions = torch.stack([step[:, :, -1, :] for step in all_steps], dim=1)  # (batch, steps, K, 2)
        goal_penalty = torch.mean((all_final_positions - goal_pos_exp) ** 2)

        if return_all_steps:
            # Stack: [initial, refine1, refine2, refine3, refine4, refine5]
            return torch.stack(all_steps, dim=1), goal_penalty  # (batch, 6, num_samples, pred_len, 2), scalar
        else:
            # Return only final refined output (for inference)
            return refined_pred, goal_penalty  # (batch, num_samples, pred_len, 2), scalar


# =============================================================================
# Main Model: HDGT with goal distribution, diffusion, refiner and ranker
# =============================================================================

def flag_on(value):
    """A flag that may arrive as a bool or as argparse's "True"/"False" string"""
    return value in (True, 'True', 'true', '1')


def straight_chains(goals, obs_pos, pred_len, use_goals):
    """The K starting chains without the diffusion decoder (--use_diffusion False), as per-step
    VELOCITIES (N, K, T, 2) like ddim_sample's: a straight line at constant speed from the last
    observed position to each of the K goals (N, K, 2); without goals (--use_goal_predictor
    False), constant velocity (the last observed step) for every chain"""
    n, k = goals.shape[:2]
    if use_goals:
        return (goals / float(pred_len)).unsqueeze(2).expand(n, k, pred_len, 2).contiguous()
    v_last = obs_pos[:, -1] - obs_pos[:, -2]                                  # (N, 2)
    return v_last[:, None, None, :].expand(n, k, pred_len, 2).contiguous()


def neighbour_context(agent_ids, input_dic, hidden_dim):
    """The tokens TransformerDiffusionDecoder.get_context_features gathers, before its
    projection: the agent's own state and its neighbours' edge states, (N, M, hidden_dim), with
    the padding mask (N, M), True = padding; (N, 1, hidden_dim) and None without neighbours.
    The ranker's context when there is no decoder (--use_diffusion False)"""
    g = input_dic["graph_lis"]
    self_features = g.nodes["agent"].data["a_n_hidden"][agent_ids]
    device = self_features.device
    context_list, max_neighbors = [], 0
    for agent_id in agent_ids:
        eids = g.in_edges(etype="other", v=agent_id, form="eid")
        if len(eids) > 0:
            context_list.append(g.edges["other"].data["a_e_hidden"][eids])
            max_neighbors = max(max_neighbors, len(eids))
        else:
            context_list.append(torch.zeros((0, hidden_dim), device=device))
    if max_neighbors == 0:
        return self_features.unsqueeze(1), None
    padded, masks = [], []
    for ctx in context_list:
        pad = max_neighbors - ctx.shape[0]
        padded.append(torch.cat([ctx, torch.zeros((pad, hidden_dim), device=device)], dim=0))
        masks.append(torch.cat([torch.zeros(ctx.shape[0], device=device),
                                torch.ones(pad, device=device)], dim=0).bool())
    context = torch.cat([self_features.unsqueeze(1), torch.stack(padded, dim=0)], dim=1)
    mask = torch.cat([torch.zeros((len(agent_ids), 1), device=device).bool(), torch.stack(masks, dim=0)], dim=1)
    return context, mask


class HDGT_model(nn.Module):
    """
    hdgt_full: HDGT with the goal distribution, the diffusion decoder, the refiner and the RankNN
    ranker, every component behind a flag (HDGT_FULL_PLAN.md). train.py's defaults in brackets.

    Components:
        encoder          HDGT heterogeneous graph network; + the map's shapes as graph nodes
                         (--use_map_nodes [True])
        goal predictor   (--use_goal_predictor [True]) heatmap: a goal distribution over a grid
                         around each agent, with a map crop and a reachability prior, the K goals
                         picked from it (--goal_predictor [heatmap]); mtp: K WTA heads. Off: no
                         goals (zeros) for the decoder, the refiner and the ranker
        diffusion        (--use_diffusion [True]) K goal-conditioned DDIM chains. Off: straight
                         lines to the K goals (constant velocity without goals)
        refiner          (--use_refiner [True]) refines the K chains
        ranker           (--use_ranker [True]) RankNN: orders the K chains, best first. Off: the
                         chains keep the goal order (most likely goal first)
        map penalties    (--use_map_penalty [False], parts --goal/diffusion/refiner_map_penalty)

    Training losses:
        diffusion  noise prediction + alpha_t-weighted goal penalty
        refiner    best-of-K distance to the GT path
        goal       heatmap: cross-entropy against a Gaussian around the GT goal; mtp: K-head WTA
        ranker     orders the K chains by ADE to GT, once the goal curriculum is done; its inputs
                   are detached, so it moves only the ranker
        map        the depth inside shapes their agent may not enter, of the goals (heatmap: the
                   expected depth under its distribution), the denoised paths and the refined
                   paths, each weighted
    """
    def __init__(self, input_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.args = args
        self.num_samples = args.num_prediction
        self.pred_len = args.pred_len
        self.obs_len = args.obs_len

        # LOSS WEIGHTS
        self.diffusion_loss_weight = 1.0
        self.refiner_loss_weight = 1.0
        self.goal_loss_weight = getattr(args, 'goal_loss_weight', 0.5)

        # The components. The getattr defaults are train.py's, so test_sdd.py and visualize_goal.py
        # rebuild the model as trained from a checkpoint's saved args
        self.use_goal_predictor = flag_on(getattr(args, 'use_goal_predictor', True))
        self.use_diffusion = flag_on(getattr(args, 'use_diffusion', True))
        self.use_refiner = flag_on(getattr(args, 'use_refiner', True))
        self.use_ranker = flag_on(getattr(args, 'use_ranker', True))

        # Goal predictor (GOAL_HEATMAP_PLAN.md)
        self.goal_predictor_type = getattr(args, 'goal_predictor', 'heatmap')
        # False: the goal loss also trains the encoder (by default it reads detached features)
        self.goal_detach_encoder = flag_on(getattr(args, 'goal_detach_encoder', True))
        # > 0: in training only, the goals the denoiser and the refiner are conditioned on get
        # Gaussian noise of RMS goal_cond_noise * s_n, so the decoder learns to trust goals as
        # much as the predicted ones deserve at test; 0 (default): the clean goals
        self.goal_cond_noise = float(getattr(args, 'goal_cond_noise', 0.0))
        self.goal_scale_min = float(getattr(args, 'goal_scale_min', 0.2))
        self.goal_scale_class_frac = float(getattr(args, 'goal_scale_class_frac', 0.5))
        # The goal distribution (--goal_predictor heatmap)
        self.goal_map_crop = flag_on(getattr(args, 'goal_map_crop', True))
        self.goal_reach_prior = flag_on(getattr(args, 'goal_reach_prior', True))
        self.goal_grid_cells = int(getattr(args, 'goal_grid_cells', 61))
        self.goal_grid_range = float(getattr(args, 'goal_grid_range', 1.5))
        self.goal_heatmap_sigma = float(getattr(args, 'goal_heatmap_sigma', 0.075))
        self.goal_crop_cells = int(getattr(args, 'goal_crop_cells', 32))
        self.goal_hidden = int(getattr(args, 'goal_hidden', 128))
        # How the K goals are picked from it (see pick_goals)
        self.goal_pick = getattr(args, 'goal_pick', 'kmeans')
        self.goal_pick_iters = int(getattr(args, 'goal_pick_iters', 25))
        self.goal_pick_alpha = float(getattr(args, 'goal_pick_alpha', 0.5))
        self.goal_pick_reserve = int(getattr(args, 'goal_pick_reserve', 4))
        self.goal_pick_reserve_alpha = float(getattr(args, 'goal_pick_reserve_alpha', 0.25))
        self.goal_pick_floor = float(getattr(args, 'goal_pick_floor', 0.5))

        # Map penalties (hdgt_map_gnn's): a goal (for the goal distribution, its mass), a denoised
        # path or a refined path inside a shape its agent may not enter (obstacles for everyone,
        # grass for all but pedestrians) costs its depth there (map_depth: linear, 0 outside),
        # times a weight in [0, 1]. --use_map_penalty switches the three together; a part whose
        # component is off does nothing. With --map_penalty_gt_exempt, the shapes an agent's own
        # GT path enters are left out for that agent, so a penalty never pushes against the GT.
        use_map_penalty = flag_on(getattr(args, 'use_map_penalty', False))
        self.goal_map_penalty = (use_map_penalty and self.use_goal_predictor
                                 and flag_on(getattr(args, 'goal_map_penalty', True)))
        self.diffusion_map_penalty = (use_map_penalty and self.use_diffusion
                                      and flag_on(getattr(args, 'diffusion_map_penalty', True)))
        self.refiner_map_penalty = (use_map_penalty and self.use_refiner
                                    and flag_on(getattr(args, 'refiner_map_penalty', True)))
        self.goal_map_penalty_weight = float(getattr(args, 'goal_map_penalty_weight', 0.1))
        self.diffusion_map_penalty_weight = float(getattr(args, 'diffusion_map_penalty_weight', 0.1))
        self.refiner_map_penalty_weight = float(getattr(args, 'refiner_map_penalty_weight', 0.1))
        self.map_penalty_gt_exempt = flag_on(getattr(args, 'map_penalty_gt_exempt', True))
        self.last_map_penalties = {}     # the switched-on penalties of the last training batch

        # The ranker (hdgt_RankNN). Its loss starts once the goal curriculum is done (at once
        # without a goal predictor): before that, every chain is conditioned on a goal blended
        # toward GT, so the order to learn is not the one inference produces
        self.ranker_loss_weight = getattr(args, 'ranker_loss_weight', 0.5)
        self.ranker_map = flag_on(getattr(args, 'ranker_map', True))
        self.ranker_image = flag_on(getattr(args, 'ranker_image', True))

        for message, ok in (
            ("--goal_predictor must be 'heatmap' or 'mtp'", self.goal_predictor_type in ('heatmap', 'mtp')),
            ("--goal_cond_noise must be >= 0", self.goal_cond_noise >= 0),
            ("--goal_scale_min must be > 0 (s_n scales the goal grid)", self.goal_scale_min > 0),
            ("--goal_scale_class_frac must be >= 0", self.goal_scale_class_frac >= 0),
            ("--goal_grid_cells must be >= 3", self.goal_grid_cells >= 3),
            ("--goal_grid_range and --goal_heatmap_sigma must be > 0",
             min(self.goal_grid_range, self.goal_heatmap_sigma) > 0),
            ("--goal_crop_cells must be >= 8", self.goal_crop_cells >= 8),
            ("--goal_hidden must be >= 1", self.goal_hidden >= 1),
            (f"--goal_pick must be one of {GOAL_PICK_METHODS}", self.goal_pick in GOAL_PICK_METHODS),
            ("--goal_pick_iters must be >= 1", self.goal_pick_iters >= 1),
            ("--goal_pick_alpha and --goal_pick_reserve_alpha must be in (0, 1]",
             0 < self.goal_pick_alpha <= 1 and 0 < self.goal_pick_reserve_alpha <= 1),
            ("--goal_pick_reserve must be in [0, K - 1]", 0 <= self.goal_pick_reserve < self.num_samples),
            ("--goal_pick_floor must be in [0, 1]", 0.0 <= self.goal_pick_floor <= 1.0),
            ("--goal/diffusion/refiner_map_penalty_weight must be in [0, 1]",
             all(0.0 <= w <= 1.0 for w in (self.goal_map_penalty_weight, self.diffusion_map_penalty_weight,
                                           self.refiner_map_penalty_weight))),
            ("--ranker_models must be >= 1", int(getattr(args, 'ranker_models', 5)) >= 1),
        ):
            if not ok:
                raise ValueError(message)
        # The goal grid's cell centres r, (G*G, 2): no parameters and nothing in the state_dict
        self.register_buffer('goal_grid', goal_grid(self.goal_grid_cells, self.goal_grid_range),
                             persistent=False)

        # Model components, in hdgt_RankNN's order; a component that is off is not built
        self.encoder = HDGT_encoder(input_dim, args)
        self.diffusion_decoder = TransformerDiffusionDecoder(args) if self.use_diffusion else None
        self.refiner = DiffusionRefiner(args) if self.use_refiner else None
        if not self.use_goal_predictor:
            self.goal_predictor = None
        elif self.goal_predictor_type == 'heatmap':
            self.goal_predictor = GoalPredictorHeatmap(
                args.hidden_dim, args.obs_len * input_dim, self.goal_grid_cells, hidden=self.goal_hidden,
                use_crop=self.goal_map_crop, use_reach=self.goal_reach_prior, crop_cells=self.goal_crop_cells,
            )
        else:
            self.goal_predictor = GoalPredictorMTP(args.hidden_dim, num_modes=self.num_samples)
        # Every ranker input is detached, so its loss cannot reward the generator for chains that
        # are easy to order, and train.py gives it its own parameter group and gradient clip
        self.ranker = build_ranker(
            args.hidden_dim, self.pred_len, self.num_samples, DDIM_N_SETTLE,
            ranker_map=self.ranker_map, ranker_image=self.ranker_image,
            image_size=int(getattr(args, 'ranker_image_size', 64)),
            image_px=float(getattr(args, 'ranker_image_px', 16.0)),
            scale=float(getattr(args, 'scale_factor', 50.0)),
            n_models=int(getattr(args, 'ranker_models', 5))) if self.use_ranker else None

        # The class statistics behind the s_n floors: no parameters and nothing in the state_dict
        # (train.py saves them in the checkpoint's extra dict; test_sdd.py loads them back)
        self.goal_class_scale = GoalClassScale(6)
        self.last_goal_stats = None
        self.last_ranker_loss = None

        # Controlled by training loop each epoch (start at 0 → ramp up)
        self.goal_curriculum_alpha = 0.0  # 0 = full oracle, 1 = full predicted goal
        # Ranker soft-pick temperature, annealed across the run (progress 0 -> 1, set by
        # train.py): soft at first, so every chain gets gradient and the whole order is learnt,
        # close to the hard argmin used at inference by the end
        self.ranker_progress = 0.0
        self.ranker_tau_pick_start = 2.0
        self.ranker_tau_pick_end = 0.3
        # The ranker's target, per-chain ADE, is in PIXELS. train.py passes the dataset's scale
        self.ranker_pixel_scale = float(getattr(args, 'scale_factor', 50.0))
        # cache_candidates-style tools set this: _rank_chains then keeps its inputs
        self.keep_rank_inputs = False

    def goal_class_floors(self):
        """The current class floors s_min(c) of the motion scale, (6,), scaled (see class_floors)"""
        return class_floors(self.goal_class_scale.scale, self.goal_class_scale.count,
                            self.goal_scale_min, self.goal_scale_class_frac)

    def _goal_logits(self, agent_feat, target_ids, target_types, input_dic, cv_end, s_n):
        """GoalPredictorHeatmap's logits (N, G*G). Its map crop and reachability features are built
        from the batch's per-target map geometry (no gradient through them)"""
        obs_feat = input_dic["graph_lis"].nodes["agent"].data["a_n_fea"][target_ids].flatten(1)
        onehot = F.one_hot(target_types.long(), 6)
        crop = reach = None
        if self.goal_map_crop or self.goal_reach_prior:
            geom, cls = input_dic["map_geom_lis"], input_dic["map_cls_lis"]
            forbid, img = input_dic["map_forbid_mask_lis"], input_dic["map_img_lis"]
            with torch.no_grad():
                s32, cv32 = s_n.float(), cv_end.float()
                if self.goal_map_crop:
                    crop = map_crop(geom, cls, forbid, img, s32, self.goal_crop_cells).to(agent_feat.dtype)
                if self.goal_reach_prior:
                    allowed = grid_allowed(geom, forbid, img, cv32, s32, self.goal_grid)
                    reach = reach_features(allowed, cv32, s32, self.goal_grid, self.goal_grid_cells,
                                           self.goal_grid_range).to(agent_feat.dtype)
        return self.goal_predictor(agent_feat, obs_feat.to(agent_feat.dtype), onehot.to(agent_feat.dtype),
                                   crop, reach)

    def _pick(self, logits, cv_end, s_n):
        """The K goals (N, K, 2) picked from the distribution softmax(logits), most likely first,
        their probabilities (N, K) and the distribution (N, G*G)"""
        prob = torch.softmax(logits.detach().float(), dim=1)
        cells, mass = pick_goals(prob, self.goal_grid, self.num_samples, method=self.goal_pick,
                                 iters=self.goal_pick_iters, alpha=self.goal_pick_alpha,
                                 reserve=self.goal_pick_reserve, reserve_alpha=self.goal_pick_reserve_alpha,
                                 floor=self.goal_pick_floor)
        goals = cv_end.float().unsqueeze(1) + s_n.float().view(-1, 1, 1) * cells
        return goals, mass, prob

    def _context(self, target_agent_ids, input_dic):
        """The ranker's neighbour tokens: the decoder's (get_context_features), or without a
        decoder the same tokens before its projection (neighbour_context)"""
        if self.use_diffusion:
            return self.diffusion_decoder.get_context_features(target_agent_ids, input_dic)
        return neighbour_context(target_agent_ids, input_dic, self.hidden_dim)

    def _rank_chains(self, diffusion_samples, all_refined_steps, goals, obs_trajectories,
                     target_agent_ids, target_agent_types, input_dic):
        """Score the K chains with every ranker (hdgt_RankNN). Returns (positions, outputs):

            positions  (N, K, T, 2)  the chains they scored, as POSITIONS relative to the last
                                     observation: the refined ones, or the starting chains'
                                     velocities summed when the refiner is off
            outputs    one (err_score, log_ade) per ranker (rankers_of), each (N, K): err_score
                       LOW is good, so sort ascending; log_ade the predicted log ADE of each
                       chain, in pixels. combine_rank_scores makes them one.

        All in slot order. The one way into the ranker, for training and inference alike. Its
        inputs are built under no_grad: the ranker learns from the generator's output and sends
        no gradient back into it. The DDIM settling diagnostic is read off the decoder (zeros
        without one); the map shapes come with every batch; the image patches when the dataset
        was built for an image ranker (sdd_dataset.rank_image_settings).
        """
        with torch.no_grad():
            if all_refined_steps is None:
                positions = torch.cumsum(diffusion_samples, dim=2)
                # zero is the honest effort for chains no refiner touched
                refine_delta = positions.new_zeros(positions.shape[0], positions.shape[1], REFINE_STEPS)
            else:
                # all_refined_steps is (N, refine_num+1, K, T, 2): how far the refiner had to drag
                # each chain at each step, (N, K, refine_num)
                positions = all_refined_steps[:, -1]
                refine_delta = (
                    (all_refined_steps[:, 1:] - all_refined_steps[:, :-1])
                    .norm(dim=-1).mean(dim=-1)      # (N, refine_num, K)
                    .transpose(1, 2).contiguous()   # (N, K, refine_num)
                )
            settle = self.diffusion_decoder.last_settle.float() if self.use_diffusion else None
            state, geom = build_rank_features(
                positions.float(), goals.float(), obs_trajectories.float(), refine_delta.float(), settle)
            if self.ranker_map:
                geom = torch.cat([geom, rank_map_features(
                    positions.float(), obs_trajectories.float(), input_dic["map_geom_lis"],
                    input_dic["map_forbid_mask_lis"], self.ranker_pixel_scale)], dim=-1)
            image = None
            if self.ranker_image:
                if "rank_image_lis" not in input_dic:
                    raise ValueError("the ranker reads the image, but the batch has no image patches: "
                                     "build SDDDataset with rank_image=rank_image_settings(args)")
                image = input_dic["rank_image_lis"]
            # Neighbour context tokens, for judging social conflict: one set per agent
            ctx, ctx_mask = self._context(target_agent_ids, input_dic)

        if self.keep_rank_inputs:
            self.last_rank_inputs = dict(state=state, geom=geom, agent_types=target_agent_types,
                                         ctx=ctx, ctx_mask=ctx_mask, positions=positions)
        outputs = [ranker(state, geom, target_agent_types, ctx, ctx_mask,
                          image=image, positions=positions.float())
                   for ranker in rankers_of(self.ranker)]
        return positions, outputs

    def forward(self, input_dic):
        """
        Forward pass for training and inference.

        Training returns: (total_loss, diffusion_loss, refiner_loss, goal_loss, goal_recon_loss,
                           type_masks, types)
        Inference returns: (trajectories, type_masks, types), the K trajectories of each agent in
                           the ranker's order, best first (without a ranker: the goal order)
        """
        # Encode scene
        output_het_graph = self.encoder(input_dic)

        neighbor_size_lis = input_dic["neighbor_size_lis"]
        all_agent_raw_traj = input_dic["graph_lis"].nodes["agent"].data["a_n_fea"][..., :2].clone()
        cumsum_neighbor_size_lis = np.cumsum(neighbor_size_lis, axis=0).tolist()
        cumsum_neighbor_size_lis = [0] + cumsum_neighbor_size_lis
        pred_num_lis = input_dic["pred_num_lis"]
        agent_node_fea = output_het_graph.nodes["agent"].data["a_n_hidden"]
        all_agent_id = input_dic["graph_lis"].nodes("agent")
        agent_n_type_lis = output_het_graph.nodes["agent"].data["a_n_type"]

        # Collect target agents (those we need to predict)
        target_agent_indices = []
        target_agent_features = []
        target_agent_ids = []
        target_agent_types = []

        for i in range(1, len(cumsum_neighbor_size_lis)):
            start_idx = cumsum_neighbor_size_lis[i-1]
            pred_num = pred_num_lis[i-1]

            indices = list(range(start_idx, start_idx + pred_num))
            target_agent_indices.extend(indices)
            target_agent_features.append(agent_node_fea[start_idx:start_idx + pred_num])
            target_agent_ids.append(all_agent_id[start_idx:start_idx + pred_num])
            target_agent_types.append(agent_n_type_lis[start_idx:start_idx + pred_num])

        target_agent_indices = torch.tensor(target_agent_indices, device=agent_node_fea.device)
        target_agent_features = torch.cat(target_agent_features, dim=0)
        target_agent_ids = torch.cat(target_agent_ids, dim=0)
        target_agent_types = torch.cat(target_agent_types, dim=0)
        # target_agent_types pick the per-type weights; the metrics read each target's true type,
        # which differs for buses with --bus_as_car (sdd_dataset)
        report_types = input_dic.get("target_type_lis", target_agent_types)

        # Create type masks for evaluation
        pred_indice_bool_type_lis = [
            report_types == agent_type_idx
            for agent_type_idx in range(6)
        ]
        n_target, n_goal = target_agent_features.shape[0], self.num_samples

        if self.training:
            # ═════════════════════════════════════════════════════════════
            # TRAINING MODE: goal + diffusion + refiner + ranker (+ map penalties)
            # ═════════════════════════════════════════════════════════════
            labels_vel = input_dic["label_lis"]      # velocities for diffusion
            labels_pos = input_dic["label_pos_lis"]  # positions for refiner
            gt_goal = input_dic["goal_pos_lis"]      # GT goal endpoint (regressor target)

            # Map penalties: the shapes each target may not enter, in its own frame
            zero = torch.zeros((), device=labels_pos.device)
            goal_map_loss = diffusion_map_loss = refiner_map_loss = zero
            map_geom = map_mask = None
            if self.goal_map_penalty or self.diffusion_map_penalty or self.refiner_map_penalty:
                map_geom = input_dic["map_geom_lis"]
                map_mask = input_dic["map_loss_mask_lis" if self.map_penalty_gt_exempt
                                     else "map_forbid_mask_lis"]

            # ─────────────────────────────────────────────────────────────
            # LOSS 3: Goal predictor (the goal distribution, or mtp's K WTA heads)
            # ─────────────────────────────────────────────────────────────
            # Detach encoder features so goal gradients don't flow into the
            # encoder (--goal_detach_encoder, default True): the goal predictor
            # reads encoder outputs; the encoder is trained by diffusion + refiner.
            goal_features = target_agent_features.detach() if self.goal_detach_encoder else target_agent_features
            gt_goal32 = gt_goal.float()

            # Motion scale s_n, with the class floors of the agents seen BEFORE this batch
            obs_trajectories = all_agent_raw_traj[target_agent_ids]
            kin_scale = kinematic_scale(obs_trajectories, self.pred_len)              # (N,)
            s_n = torch.max(kin_scale, self.goal_class_floors()[target_agent_types])  # (N,)

            goal_loss = goal_recon_loss = zero
            if self.use_goal_predictor:
                if self.goal_predictor_type == 'heatmap':
                    # Every agent trains every cell: cross-entropy against a Gaussian around its true
                    # goal. The K goals are then picked from the distribution, without gradient
                    cv_end = cv_endpoint(obs_trajectories, self.pred_len)
                    heat_logits = self._goal_logits(goal_features, target_agent_ids, target_agent_types,
                                                    input_dic, cv_end, s_n).float()
                    heat_target = goal_soft_target((gt_goal32 - cv_end.float()) / s_n.float().unsqueeze(-1),
                                                   self.goal_grid, self.goal_heatmap_sigma, self.goal_grid_range)
                    heat_log_p = F.log_softmax(heat_logits, dim=1)
                    goal_assign_loss = -(heat_target * heat_log_p).sum(dim=1).mean()
                    predicted_goals, _, _ = self._pick(heat_logits, cv_end, s_n)
                    if self.goal_map_penalty:
                        # The expected depth of a goal under the distribution: every cell's depth
                        # inside a shape the agent may not enter, weighted by its probability
                        cells = (cv_end.float().unsqueeze(1)
                                 + s_n.float().view(-1, 1, 1) * self.goal_grid.unsqueeze(0))   # (N, G*G, 2)
                        cell_depth = map_depth(cells, map_geom, map_mask)                     # (N, G*G)
                        goal_map_loss = (heat_log_p.exp() * cell_depth).sum(dim=1).mean()
                else:
                    predicted_goals = self.goal_predictor(goal_features)    # (N, K, 2)

                # Per-goal squared distance to the GT endpoint, per coordinate (scaled^2 units)
                goals32 = predicted_goals.float()
                goal_dists = ((goals32 - gt_goal32.unsqueeze(1)) ** 2).mean(dim=-1)  # (N, K)
                winner_idx = goal_dists.argmin(dim=1)  # (N,)
                num_heads = goal_dists.shape[1]
                if self.goal_predictor_type == 'mtp':
                    # hdgt_goal's loss exactly. Relaxed WTA: while the curriculum is still ramping
                    # (alpha < 1), spread the regression loss over the top-3 closest heads; afterwards,
                    # pure top-1 winner-takes-all. No confidence head: the ranker orders the chains
                    if self.goal_curriculum_alpha < 1.0:
                        goal_assign_loss = goal_dists.topk(min(3, num_heads), dim=1, largest=False).values.mean()
                    else:
                        goal_assign_loss = goal_dists.gather(1, winner_idx.unsqueeze(1)).mean()
                    if self.goal_map_penalty:     # every head's goal point, inside a forbidden shape
                        goal_map_loss = map_depth(predicted_goals, map_geom, map_mask).mean()
                goal_loss = goal_assign_loss
                # Logged separately: best-goal MSE — the interpretable goal quality
                # (pixel RMSE = sqrt(value) * 50). Occupies the old goal_recon slot.
                goal_recon_loss = goal_dists.gather(1, winner_idx.unsqueeze(1)).mean().detach()
                self.last_goal_stats = {"assign": goal_assign_loss.detach()}

                # Goal curriculum, two forms:
                # (a) winner blend (N, 2) for the diffusion noise-prediction loss —
                #     the GT trajectory's endpoint ~= winner goal, so conditioning
                #     the denoiser on this blend stays consistent with its label.
                winner_goal = predicted_goals.gather(
                    1, winner_idx.view(-1, 1, 1).expand(-1, 1, 2)
                ).squeeze(1)  # (N, 2)
                effective_goal = (
                    self.goal_curriculum_alpha * winner_goal.detach() +
                    (1.0 - self.goal_curriculum_alpha) * gt_goal
                )
                # (b) per-chain blend (N, K, 2) for ddim_sample + refiner — chain k
                #     is conditioned on goal k. At alpha=0 every chain sees GT; at
                #     alpha=1 each chain sees its own goal, exactly matching inference.
                effective_goals_k = (
                    self.goal_curriculum_alpha * predicted_goals.detach() +
                    (1.0 - self.goal_curriculum_alpha) * gt_goal.unsqueeze(1)
                )
                # --goal_cond_noise: the denoiser and the refiner are conditioned on these goals plus
                # Gaussian noise of RMS goal_cond_noise * s_n; the denoiser's endpoint penalty keeps
                # the clean goal as its target. Off (0, default): the clean goals
                cond_goal, cond_goals_k = effective_goal, effective_goals_k
                if self.goal_cond_noise > 0:
                    noise_scale = (self.goal_cond_noise / math.sqrt(2.0)) * s_n   # (N,), per coordinate
                    cond_goal = effective_goal + noise_scale[:, None] * torch.randn_like(effective_goal)
                    cond_goals_k = effective_goals_k + noise_scale[:, None, None] * torch.randn_like(effective_goals_k)
            else:
                # No goal predictor: no goals anywhere, in training as at inference
                effective_goal = cond_goal = gt_goal.new_zeros((n_target, 2))
                cond_goals_k = gt_goal.new_zeros((n_target, n_goal, 2))
                self.last_goal_stats = {}

            # Only now, after this batch's loss, do its agents enter the class scales: no agent's
            # own GT sets its floor
            self.goal_class_scale.update(gt_goal32, target_agent_types)

            # ─────────────────────────────────────────────────────────────
            # LOSS 1: Noise prediction + alpha_t-weighted goal penalty
            # ─────────────────────────────────────────────────────────────
            diffusion_loss = zero
            if self.use_diffusion:
                predicted_noise, actual_noise, goal_penalty, diffusion_map_loss = self.diffusion_decoder.training_forward(
                    target_agent_features,
                    target_agent_ids,
                    target_agent_types,
                    labels_vel,
                    cond_goal,
                    input_dic,
                    goal_target=effective_goal if self.goal_cond_noise > 0 else None,
                    use_goal_penalty=self.use_goal_predictor,
                    map_geom=map_geom if self.diffusion_map_penalty else None,
                    map_mask=map_mask,
                )
                diffusion_loss = F.mse_loss(predicted_noise, actual_noise) + goal_penalty

            # ─────────────────────────────────────────────────────────────
            # LOSS 2: Multi-Step Refiner, on the K chains
            # ─────────────────────────────────────────────────────────────
            # The K chains are drawn whenever the refiner or the ranker needs them: DDIM chains,
            # or without the decoder straight lines to the goals. Recording the settling
            # diagnostic (a ranker input) changes no chain.
            refiner_loss = zero
            diffusion_samples = all_refined_steps = None
            if self.use_refiner or self.use_ranker:
                with torch.no_grad():
                    if self.use_diffusion:
                        diffusion_samples = self.diffusion_decoder.ddim_sample(
                            target_agent_features,
                            target_agent_ids,
                            target_agent_types,
                            cond_goals_k,   # (N, K, 2): chain k conditioned on goal k
                            input_dic,
                            num_samples=self.num_samples,
                            ddim_steps=20,
                            collect_settle=self.use_ranker,
                        )
                    else:
                        diffusion_samples = straight_chains(cond_goals_k, obs_trajectories, self.pred_len,
                                                            self.use_goal_predictor)

            if self.use_refiner:
                all_refined_steps, _ = self.refiner(
                    diffusion_samples,
                    obs_trajectories,
                    target_agent_features,
                    cond_goals_k,   # (N, K, 2): per-chain goal conditioning
                    return_all_steps=True
                )

                label_positions_exp = labels_pos.unsqueeze(1).unsqueeze(1).expand(
                    -1, all_refined_steps.shape[1], self.num_samples, -1, -1
                )
                distances = torch.sqrt(
                    ((all_refined_steps - label_positions_exp) ** 2).sum(dim=-1).mean(dim=-1)
                )  # (batch, 6_steps, K_samples)

                best_mode_idx = distances[:, -1, :].argmin(dim=-1)
                best_mode_distances = distances[
                    torch.arange(distances.shape[0]),
                    :,
                    best_mode_idx
                ]  # (batch, 6_steps)

                # No refiner goal penalty: best_mode_distances already supervises the
                # full trajectory (endpoint included) against the GT trajectory, which
                # is strictly more accurate than pushing the endpoint toward a predicted
                # goal. The goal still conditions the refiner as an input.
                refiner_loss = best_mode_distances.mean()

                if self.refiner_map_penalty:   # every refined chain, densified
                    refiner_map_loss = map_depth(
                        densify_path(all_refined_steps[:, -1]), map_geom, map_mask).mean()

            # ─────────────────────────────────────────────────────────────
            # LOSS 4: ranker — rank the chains we actually select from
            # ─────────────────────────────────────────────────────────────
            # It scores the very chains the refiner loss was just computed on, each beside the
            # goal it was conditioned on, and learns to order them by per-chain ADE, the SAME
            # definition the report ranks on.
            ranker_term = zero
            if self.use_ranker:
                positions, outputs = self._rank_chains(
                    diffusion_samples, all_refined_steps, cond_goals_k,
                    obs_trajectories, target_agent_ids, target_agent_types, input_dic)
                with torch.no_grad():
                    ade_k = ((positions - labels_pos.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
                             * self.ranker_pixel_scale)                    # (N, K), pixels
                tau_pick = self.ranker_tau_pick_start + (
                    self.ranker_tau_pick_end - self.ranker_tau_pick_start) * self.ranker_progress
                # Each ranker learns from its own loss, as if trained alone (see RankEnsemble).
                # Their mean keeps the term on one ranker's scale; train.py clips each ranker's
                # gradient to match.
                ranker_loss = torch.stack([
                    rank_loss(err_score.float(), log_ade.float(), ade_k.float(),
                              target_agent_types, tau_pick=tau_pick)
                    for err_score, log_ade in outputs]).mean()
                # Gate with a float rather than skipping the term, so the ranker's parameters
                # still receive (zero) gradient and DDP does not need find_unused_parameters
                gate = 1.0 if (not self.use_goal_predictor or self.goal_curriculum_alpha >= 1.0) else 0.0
                ranker_term = gate * self.ranker_loss_weight * ranker_loss
                # Stashed UNGATED so the log shows the ranker's true state
                self.last_ranker_loss = ranker_loss.detach()

            # ─────────────────────────────────────────────────────────────
            # TOTAL LOSS
            # ─────────────────────────────────────────────────────────────
            # ranker_term only moves the ranker (its inputs are detached). Switched-off map
            # penalties are 0 here, so they add nothing
            map_term = (self.goal_map_penalty_weight * goal_map_loss +
                        self.diffusion_map_penalty_weight * diffusion_map_loss +
                        self.refiner_map_penalty_weight * refiner_map_loss)
            self.last_map_penalties = {part: value.detach() for part, value, on in (
                ("goal", goal_map_loss, self.goal_map_penalty),
                ("diffusion", diffusion_map_loss, self.diffusion_map_penalty),
                ("refiner", refiner_map_loss, self.refiner_map_penalty)) if on}
            total_loss = (
                self.diffusion_loss_weight * diffusion_loss +
                self.refiner_loss_weight * refiner_loss +
                self.goal_loss_weight * goal_loss +
                ranker_term +
                map_term
            )

            return total_loss, diffusion_loss, refiner_loss, goal_loss, goal_recon_loss, pred_indice_bool_type_lis, report_types

        else:
            # ═════════════════════════════════════════════════════════════
            # INFERENCE MODE: K goals, one per chain; the ranker orders the K chains
            # ═════════════════════════════════════════════════════════════
            obs_trajectories = all_agent_raw_traj[target_agent_ids]
            self.last_target_obs = obs_trajectories.detach()   # observed positions, for s_n
            self.last_goal_prob = None
            goal_logits = None
            if not self.use_goal_predictor:
                predicted_goals = target_agent_features.new_zeros((n_target, n_goal, 2))
            elif self.goal_predictor_type == 'heatmap':
                # The distribution, then its K goals, most likely first; their log-probabilities
                # stand in for the confidence logits (index 0 = the most likely goal)
                s_n = torch.max(kinematic_scale(obs_trajectories, self.pred_len),
                                self.goal_class_floors()[target_agent_types])
                cv_end = cv_endpoint(obs_trajectories, self.pred_len)
                heat_logits = self._goal_logits(target_agent_features.detach(), target_agent_ids,
                                                target_agent_types, input_dic, cv_end, s_n)
                predicted_goals, goal_mass, goal_prob = self._pick(heat_logits, cv_end, s_n)
                goal_logits = goal_mass.clamp_min(1e-12).log()
                # For test_sdd.py's GOAL DISTRIBUTION block
                self.last_goal_prob = goal_prob.detach()
                self.last_goal_cv = cv_end.detach()
                self.last_goal_s = s_n.detach()
            else:
                predicted_goals = self.goal_predictor(target_agent_features.detach())  # (batch, K, 2)
            # The goals in slot order (slot k = chain k), before the ranker permutes the chains:
            # the GOAL HEAD diagnostics read these
            self.last_goals_head_order = predicted_goals.detach()
            self.last_goal_logits = goal_logits.detach() if goal_logits is not None else None

            # The K chains: DDIM, or straight lines to the goals without the decoder
            if self.use_diffusion:
                diffusion_samples = self.diffusion_decoder.ddim_sample(
                    target_agent_features,
                    target_agent_ids,
                    target_agent_types,
                    predicted_goals,
                    input_dic,
                    num_samples=self.num_samples,
                    ddim_steps=20,
                    collect_settle=self.use_ranker,   # a ranker input
                )
            else:
                diffusion_samples = straight_chains(predicted_goals, obs_trajectories, self.pred_len,
                                                    self.use_goal_predictor)

            trajectories = diffusion_samples
            all_refined_steps = None
            if self.use_refiner:
                if self.use_ranker:
                    # return_all_steps hands the ranker the refiner's per-step effort; the final
                    # chains are the same either way
                    all_refined_steps, _ = self.refiner(
                        diffusion_samples, obs_trajectories, target_agent_features, predicted_goals,
                        return_all_steps=True)
                    trajectories = all_refined_steps[:, -1]
                else:
                    trajectories, _ = self.refiner(
                        diffusion_samples, obs_trajectories, target_agent_features, predicted_goals,
                        return_all_steps=False)

            # Rank the K chains so index 0 is the one we predict is best. A pure output
            # permutation AFTER generation: minADE/minFDE are unaffected, while single-mode
            # ADE/FDE (preds[:, 0]) come from the ranking. Without a ranker the chains keep the
            # goal order (heatmap: most likely goal first).
            self.last_pred_error = None
            if self.use_ranker:
                _, outputs = self._rank_chains(
                    diffusion_samples, all_refined_steps, predicted_goals,
                    obs_trajectories, target_agent_ids, target_agent_types, input_dic)
                err_score, log_ade = combine_rank_scores(outputs)
                # err_score is LOW-is-good, so ascending
                conf_order = torch.argsort(err_score, dim=1, descending=False)
                # The ranker's absolute error estimate for each chain, in the returned order, PIXELS
                self.last_pred_error = torch.exp(log_ade.gather(1, conf_order)).detach()
                trajectories = trajectories.gather(
                    1,
                    conf_order[:, :, None, None].expand(
                        -1, -1, trajectories.shape[2], trajectories.shape[3]
                    )
                )
                predicted_goals = predicted_goals.gather(
                    1, conf_order[:, :, None].expand(-1, -1, 2))

            # The goals in the same order as the returned chains (goal k belongs to chain k)
            self.last_predicted_goals = predicted_goals.detach()

            self.output_is_positions = self.use_refiner

            return trajectories, pred_indice_bool_type_lis, report_types


# =============================================================================
# GOAL HEAD diagnostics: train.py's validation summary and test_sdd.py's report
# =============================================================================

def goal_head_agent_arrays(goals, gt_goal, s, scale_factor, first_goal=None, conf_logits=None,
                           rho=0.15):
    """Per-trajectory quantities of the GOAL HEAD diagnostics (numpy). goals (n, K, 2): the K
    heads in head order, before any ranking; gt_goal (n, 2): the true goal; s (n,): the motion
    scale s_n; all scaled, agent frame. first_goal (n, 2): the goal of the first returned chain;
    conf_logits (n, K): the confidence head's, when the model has one. Every distance here is
    Euclidean, so checkpoints compare on one footing"""
    goals, gt_goal, s = (np.asarray(x, dtype=np.float64) for x in (goals, gt_goal, s))
    n, K, _ = goals.shape
    err = np.linalg.norm(goals - gt_goal[:, np.newaxis], axis=-1)            # (n, K), scaled
    i, j = np.triu_indices(K, k=1)
    pair = np.linalg.norm(goals[:, i] - goals[:, j], axis=-1)                # (n, K (K - 1) / 2)
    off_radius = GOAL_OFF_FACTOR * np.maximum(s, np.linalg.norm(gt_goal, axis=-1))
    return {
        "err": err * scale_factor,                                            # px
        "winner": err.argmin(axis=1),
        "best": err.min(axis=1) * scale_factor,
        "mean_k": err.mean(axis=1) * scale_factor,
        "first": (None if first_goal is None else
                  np.linalg.norm(np.asarray(first_goal, dtype=np.float64) - gt_goal, axis=-1) * scale_factor),
        "conf": (None if conf_logits is None else
                 err[np.arange(n), np.asarray(conf_logits).argmax(axis=1)] * scale_factor),
        "min_pair": pair.min(axis=1) / s if K > 1 else np.full(n, np.nan),
        "close": (pair < rho * s[:, np.newaxis]).mean(axis=1) if K > 1 else np.full(n, np.nan),
        "off": (err > off_radius[:, np.newaxis]).mean(axis=1),
    }


def goal_head_summary(arrays, idx=None):
    """GOAL HEAD metrics of the trajectories idx (default: all), from goal_head_agent_arrays:
    numbers (None where the model has no such output) plus the per-head win rates and errors"""
    take = (lambda x: x) if idx is None else (lambda x: x[idx])
    err, best = take(arrays["err"]), take(arrays["best"])
    n, K = err.shape
    win = np.bincount(take(arrays["winner"]), minlength=K) / n
    p = win[win > 0]
    entropy = -(p * np.log(p)).sum() + 0.0    # + 0.0: no "-0.000" when one head wins every time
    slot_err = err.mean(axis=0)
    optional = {key: None if arrays[key] is None else take(arrays[key]).mean() for key in ("first", "conf")}
    return {
        "n": n,
        "entropy": entropy / np.log(K) if K > 1 else np.nan,
        "eff_k": np.exp(entropy),
        "win_min": win.min() * 100, "win_max": win.max() * 100,
        "slot_best": slot_err.min(), "slot_worst": slot_err.max(),
        "best": best.mean(), "mean_k": take(arrays["mean_k"]).mean(), **optional,
        "best_p95": np.percentile(best, 95), "all_p95": np.percentile(err, 95),
        "min_pair": take(arrays["min_pair"]).mean(), "close": take(arrays["close"]).mean() * 100,
        "off": take(arrays["off"]).mean() * 100,
        "win_rates": win * 100, "slot_err": slot_err,
    }


def goal_head_diagnostics(goals_head_order, gt_goal, s, cls, type_names, scale_factor,
                          first_goal=None, conf_logits=None, rho=0.15):
    """GOAL HEAD metrics of every trajectory ("ALL") and of each agent type present:
    {name: goal_head_summary}. cls (n,) holds type indices 0-5, in type_names' order"""
    arrays = goal_head_agent_arrays(goals_head_order, gt_goal, s, scale_factor, first_goal, conf_logits, rho)
    out = {"ALL": goal_head_summary(arrays)}
    for c, name in enumerate(type_names):
        idx = np.flatnonzero(np.asarray(cls) == c)
        if len(idx) > 0:
            out[name] = goal_head_summary(arrays, idx)
    return out


def goal_head_lines(diagnostics, type_names, floors_px, rho, pred_len, first_desc):
    """The GOAL HEAD block of the validation summary and of the test report. diagnostics from
    goal_head_diagnostics; floors_px: the six class floors of s_n, in pixels"""
    all_ = diagnostics["ALL"]
    K = len(all_["win_rates"])
    names = [name for name in ["ALL"] + list(type_names) if name in diagnostics]

    def num(value, width, digits):
        return f"{'--':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

    def table(title, columns):   # columns: (header, key, width, digits)
        rows = [f"  {title}", f"  {'Type':<12}{'N':>6}" + "".join(f"{h:>{w}}" for h, _, w, _ in columns)]
        for name in names:
            d = diagnostics[name]
            rows.append(f"  {name:<12}{d['n']:>6}" + "".join(num(d[key], w, dg) for _, key, w, dg in columns))
        return rows + [""]

    lines = [
        f"GOAL HEAD  ({K} goal heads in head order, before any ranking; goal = step {pred_len} = "
        f"{pred_len * 0.4:.1f}s, pixels)",
        "-" * 80,
        f"  win       : % of trajectories whose true goal a head is the closest to (chance {100 / K:.1f}%)",
        "  entropy   : of the win rates, 1 = every head equally often; effK = exp(entropy) = heads in use",
        "  head err  : a head's mean goal error over the trajectories; the best and the worst head",
        "  best-of-K : error of the closest head   mean-of-K: mean over the K heads (scatter)",
        f"  first     : goal of the first returned chain ({first_desc})",
        "  conf      : the confidence head's top goal (only without a scorer)",
        "  p95       : 95th percentile of best-of-K, and of every head's error",
        f"  minPair/s : smallest distance between two heads / s_n   close: % of head pairs nearer than "
        f"{rho:g} s_n",
        f"  off       : % of heads farther than {GOAL_OFF_FACTOR:g} x max(s_n, |true goal|) from the true goal",
        f"  s_n       : max(3-step mean speed, last-step speed) x {pred_len} steps, at least the class floor",
        "  floors    : " + ", ".join(f"{name} {floor:.1f}" for name, floor in zip(type_names, floors_px)) + " px",
        "",
    ]
    lines += table("Head usage and per-head goal error (px)", [
        ("entropy", "entropy", 9, 3), ("effK", "eff_k", 8, 2), ("minWin%", "win_min", 9, 2),
        ("maxWin%", "win_max", 9, 2), ("bestHead", "slot_best", 10, 2), ("worstHead", "slot_worst", 11, 2)])
    lines += table("Goal error (px)", [
        ("best-of-K", "best", 11, 2), ("mean-of-K", "mean_k", 11, 2), ("first", "first", 9, 2),
        ("conf", "conf", 9, 2), ("p95 best", "best_p95", 10, 2), ("p95 all", "all_p95", 10, 2)])
    lines += table("Collapse and off goals", [
        ("minPair/s", "min_pair", 11, 3), ("close%", "close", 9, 2), ("off%", "off", 9, 2)])
    lines.append("  Per head, ALL (win %, mean goal error px):")
    for start in range(0, K, 10):
        heads = range(start, min(start + 10, K))
        lines += ["    head" + "".join(f"{k:>7d}" for k in heads),
                  "    win%" + "".join(f"{all_['win_rates'][k]:>7.2f}" for k in heads),
                  "    err " + "".join(f"{all_['slot_err'][k]:>7.1f}" for k in heads)]
    lines.append("=" * 80)
    return lines


# =============================================================================
# Utilities
# =============================================================================

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0


def weights_init(m):
    """Initialize weights (skip sinusoidal embeddings)"""
    classname = m.__class__.__name__
    if 'Sinusoidal' in classname:
        return

    if hasattr(m, 'weight') and m.weight is not None:
        if classname.find('Conv1d') != -1:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif classname.find('Linear') != -1:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('GroupNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
