"""
model.py  (hdgt_goal_heatmap, see GOAL_HEATMAP_PLAN.md)
========

The full HDGT model (heterogeneous graph encoder, goal-conditioned diffusion decoder, multi-step
refiner, optional chain scorer) in which the goal predictor is a GOAL DISTRIBUTION:

    [encoder feature (detached), observed track, type] + map crop + reachability prior
        -> GoalPredictorHeatmap -> logits over a G x G grid of goals around each agent's
           constant-velocity endpoint, in units of its motion scale s_n
        -> pick_goals -> K goals, most likely first; DDIM chain k is conditioned on goal k

Every agent trains the whole grid (cross-entropy against a Gaussian around its true goal): no
heads, no assignment, no dead heads. How the K goals are picked from the distribution is
--goal_pick (kmeans | flat | reserve | sinkhorn). The map reaches the goal predictor as a crop of
the drawn shapes (grass, obstacle, forbidden for the agent, inside the image) and as travel
distances around forbidden shapes (--goal_map_crop, --goal_reach_prior); with --use_map_nodes the
shapes are also graph nodes the agents attend to in every encoder layer (as in hdgt_map_gnn).

--goal_predictor mtp keeps hdgt_goal's K-head WTA predictor (GoalPredictorMTP: relaxed top-3
during the curriculum ramp, then top-1; without a scorer + 0.5 x cross-entropy of a confidence
head on the winner), so older checkpoints load and run unchanged.

Ranking the K refined chains (index 0 is what single-trajectory metrics read):
  - --use_scorer False: no scorer; the chains keep the goal order (most likely goal first)
  - --use_scorer True --scorer_type mode | traj: ModeScorer / TrajScorer; their loss starts once
    the goal curriculum is done, and every scorer input is detached

Training conditioning (the goal curriculum, alpha 0 -> 1 over --goal_curriculum_epochs):
  - diffusion training_forward: the goal closest to the GT blended with the GT goal
  - ddim_sample + refiner: the K goals, each blended with the GT goal; at alpha=1 this is exactly
    the inference conditioning
--goal_cond_noise (off by default) would condition both on goals with Gaussian noise of RMS
sigma * s_n (the denoiser's endpoint penalty keeps the clean goal).

The main class is `HDGT_model`, so train.py / test_sdd.py (model_module.HDGT_model) are unchanged.
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
    Multimodal goal predictor (MTP-style): K candidate endpoints + confidences.

        agent_features (N, hidden) -> goals (N, K, 2), logits (N, K)

    A shared MLP trunk feeds two linear heads. Multimodality is enforced by the
    winner-takes-all loss (computed in HDGT_model.forward), which lets each
    of the K heads specialize on a distinct future mode instead of averaging
    them. No latent, no KL — nothing can collapse.

    `with_conf_head` controls the legacy blind confidence head. When a scorer
    is in use it is built with with_conf_head=False, so the trunk that
    produces the goals receives gradient ONLY from the goal regression terms.
    That matters: in scaled units the regression MSE is ~1e-2 while a 20-way
    cross-entropy is ~2-3, so a conf head on this shared trunk trains the trunk
    almost entirely for ranking and starves the regression it is sharing.
    """
    def __init__(self, hidden_dim, num_modes, with_conf_head=True):
        super().__init__()
        self.num_modes = num_modes
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.goal_head = nn.Linear(hidden_dim, num_modes * 2)
        self.conf_head = nn.Linear(hidden_dim, num_modes) if with_conf_head else None

    def forward(self, agent_features):
        h = self.trunk(agent_features)
        goals = self.goal_head(h).view(-1, self.num_modes, 2)  # (N, K, 2)
        logits = self.conf_head(h) if self.conf_head is not None else None  # (N, K)
        return goals, logits


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


class ModeScorer(nn.Module):
    """
    Permutation-equivariant scorer over the K generated candidates.

    Replaces the blind `conf_head`, which had three defects:
      1. it was trained on the GOAL winner (argmin |g_k - gt_goal|) but used at
         inference to rank TRAJECTORIES — and chain k drifts from goal k, has
         its own DDIM seed, and argmin-FDE != argmin-ADE;
      2. logit k was `u_k . h + d_k`, so it never saw the candidate it scores;
         slot k had to be recognised by a slot-specific linear detector even
         though slot semantics drift across agent types and speeds;
      3. hard cross-entropy over ~5 near-tied plausible modes converges to the
         posterior (~0.2 each), so the top of the ranking is decided by noise.

    Here every mode is scored from its OWN content with weights SHARED across
    modes, and a Deep-Sets mean-subtraction couples the modes so the score is
    comparative — ranking is inherently a relative judgement. Two outputs:

        logit_k   -> softmax ranking, trained on the trajectory winner
        log_err_k -> predicted log ADE of chain k, trained on ALL K modes

    The regression head is what gives every mode a gradient (cross-entropy only
    ever supervises one) and what gives an ABSOLUTE, calibrated score; a softmax
    over K is relative-within-agent by construction and cannot provide one.

    Every input is detached: this module cannot move the encoder, the goal
    predictor, the denoiser or the refiner.
    """

    # per-mode features: goal(2) + goal-consensus(2) + goal-ballistic(2)
    #                    + endpoint(2) + drift(2) + spread(1) + path length(1)
    N_GEOM = 12

    def __init__(self, hidden_dim, pred_len, d_model=64, dropout=0.0):
        super().__init__()
        self.pred_len = pred_len
        n_in = self.N_GEOM + 2 * pred_len          # + the path shape
        self.in_proj = nn.Linear(n_in, d_model)
        self.ctx_proj = nn.Linear(hidden_dim, d_model)
        self.mix = nn.Linear(2 * d_model, d_model)
        self.out_proj = nn.Linear(d_model, 2)      # [logit, log_err]
        # Dropout on the two hidden layers (--scorer_dropout); none by default
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @staticmethod
    def build_features(goals, trajs, obs_traj, pred_len):
        """
        Args:
            goals:    (N, K, 2)  candidate goals, agent frame, scaled
            trajs:    (N, K, T, 2) candidate POSITIONS, relative to last obs
            obs_traj: (N, obs_len, 2) observed positions, same frame
        Returns:
            (N, K, N_GEOM + 2T) per-mode feature tensor
        """
        n, k, t, _ = trajs.shape
        v0 = (obs_traj[:, -1, :] - obs_traj[:, -2, :]).unsqueeze(1)   # (N,1,2)
        ballistic = v0 * float(pred_len)                              # constant-velocity endpoint
        consensus = goals.mean(dim=1, keepdim=True)                   # (N,1,2)
        end = trajs[:, :, -1, :]                                      # (N,K,2)

        prev = torch.cat([torch.zeros_like(trajs[:, :, :1, :]), trajs[:, :, :-1, :]], dim=2)
        path_len = (trajs - prev).norm(dim=-1).sum(dim=-1, keepdim=True)   # (N,K,1)
        spread = (goals - consensus).norm(dim=-1, keepdim=True)            # (N,K,1)
        # path shape, endpoint-centred so it carries curvature not position
        shape = (trajs - end.unsqueeze(2)).reshape(n, k, t * 2)

        return torch.cat([
            goals,                    # where this mode wants to go
            goals - consensus,        # how far it sits from the consensus of the K heads
            goals - ballistic,        # how far it deviates from constant velocity
            end,                      # where the chain actually ended up
            end - goals,              # goal-following drift — invisible to the old head
            spread,
            path_len,
            shape,
        ], dim=-1)

    def forward(self, agent_features, goals, trajs, obs_traj):
        feat = self.build_features(
            goals.detach(), trajs.detach(), obs_traj.detach(), self.pred_len
        )
        h = self.drop(torch.relu(self.in_proj(feat) + self.ctx_proj(agent_features.detach()).unsqueeze(1)))
        pooled = h.mean(dim=1, keepdim=True).expand_as(h)
        h = self.drop(torch.relu(self.mix(torch.cat([h, pooled], dim=-1))))
        out = self.out_proj(h)                       # (N, K, 2)
        return out[..., 0], out[..., 1]              # logits, log_err


class TrajScorer(nn.Module):
    """
    Candidate-aware confidence scorer (fix 3): ranks the K generated chains.

        logits_k = score(trajectory_k, kinematics, agent features, neighbors)

    Unlike the old blind conf head (K logits from agent features alone, never
    seeing what it ranked), this scores each candidate explicitly:
      - per-candidate embedding of the full trajectory + endpoint + ballistic
        residual (endpoint minus constant-velocity extrapolation = "kinematic
        surprise" of this candidate)
      - fused with the agent feature
      - self-attention ACROSS the K candidates (ranking is relative — letting
        candidates see each other makes the score listwise)
      - cross-attention to neighbor context (a chain that cuts through a
        neighbor can be ranked down)
    All inputs are detached upstream: confidence never trains the encoder,
    decoder, or refiner. At inference it scores the refined trajectories; the
    same module can later score bridge prior means mu_k (they are trajectories
    too), which keeps it compatible with confidence-modulated tube width.
    """
    def __init__(self, hidden_dim, pred_len, dropout=0.1):
        super().__init__()
        d = hidden_dim
        in_dim = pred_len * 2 + 4  # flat trajectory + endpoint (2) + ballistic residual (2)
        self.cand_embed = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )
        self.agent_proj = nn.Linear(d, d)
        self.self_norm = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, 4, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, 4, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, traj, v0, agent_feat, context, context_mask):
        """
        traj: (N, K, T, 2) candidate POSITION trajectories (agent frame, scaled)
        v0:   (N, 2) last observed per-frame velocity
        agent_feat: (N, H); context: (N, C, H); context_mask: (N, C) bool or None
        returns logits (N, K)
        """
        N, K, T, _ = traj.shape
        endpoint = traj[:, :, -1, :]                        # (N, K, 2)
        ballistic = endpoint - (T * v0).unsqueeze(1)        # (N, K, 2)
        x = torch.cat([traj.reshape(N, K, T * 2), endpoint, ballistic], dim=-1)
        h = self.cand_embed(x) + self.agent_proj(agent_feat).unsqueeze(1)
        hn = self.self_norm(h)
        a, _ = self.self_attn(hn, hn, hn)
        h = h + a
        a, _ = self.cross_attn(self.cross_norm(h), context, context,
                               key_padding_mask=context_mask)
        h = h + a
        h = h + self.ffn(h)
        return self.out(h).squeeze(-1)                      # (N, K)


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
                         goal_target=None):
        """Training forward pass with noise prediction and alpha_t-weighted goal loss.

        The model predicts noise (epsilon). For the goal penalty, we recover
        x0 (velocities) from the noise prediction, convert to positions via
        cumsum, and compare the final position to the goal. The penalty is
        weighted by alpha_t so it only applies at low noise levels where x0
        recovery is accurate. goal_pos conditions the denoiser; goal_target, when
        given, replaces it as the penalty's target (--goal_cond_noise: the
        denoiser sees a noisy goal, the penalty keeps the clean one).
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
        target = goal_pos if goal_target is None else goal_target
        goal_penalty = torch.mean(alpha_t * (final_pos - target) ** 2)

        return predicted_noise, noise, goal_penalty

    @torch.no_grad()
    def ddim_sample(self, agent_features, agent_ids, agent_types, goal_pos, input_dic, num_samples=None, ddim_steps=20):
        """Fast DDIM sampling with noise prediction and goal conditioning."""
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

        for i, t_idx in enumerate(timesteps):
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
# Main Model: HDGT with Diffusion + Refiner
# =============================================================================

def flag_on(value):
    """A flag that may arrive as a bool or as argparse's "True"/"False" string"""
    return value in (True, 'True', 'true', '1')


class HDGT_model(nn.Module):
    """
    HDGT model with diffusion decoder and refiner.

    Architecture:
        1. Encoder: HDGT-style heterogeneous graph network
        2. Diffusion Decoder: Generates diverse K samples (noise prediction)
        3. Refiner: Polishes samples for accuracy (5 iterations)

    Training Losses (3-LOSS):
        1. Diffusion Loss: noise prediction + alpha_t-weighted goal penalty
        2. Multi-Step Refiner Loss: best-of-K distance to GT (no goal penalty)
        3. Goal Loss: cross-entropy of the goal distribution against a Gaussian around the GT
           goal (--goal_predictor heatmap); --goal_predictor mtp is hdgt_goal's K-head WTA
           regression (+ confidence cross-entropy without a scorer)
        4. Scorer Loss: ranks the K refined chains (ModeScorer or TrajScorer), weight
           --scorer_loss_weight, once the goal curriculum is done (off by default here)
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

        self.use_refiner = getattr(args, 'use_refiner', True)

        # The scorer ranks the K refined chains, so it needs them: it is only
        # available with the refiner on (that branch generates candidates during
        # training). --scorer_type picks ModeScorer or TrajScorer. With
        # --use_scorer False (the default in this version) the model is just
        # goal-aware, as in hdgt_goal_WTA_regressor: the goal predictor keeps its
        # confidence head and the chains stay in the goal heads' order.
        self.use_scorer = flag_on(getattr(args, 'use_scorer', False)) and self.use_refiner
        self.scorer_type = getattr(args, 'scorer_type', 'mode')
        if self.scorer_type not in ('mode', 'traj'):
            raise ValueError(f"--scorer_type must be 'mode' or 'traj', got {self.scorer_type!r}")
        scorer_dropout = args.dropout if flag_on(getattr(args, 'scorer_dropout', False)) else 0.0
        self.scorer_loss_weight = getattr(args, 'scorer_loss_weight', 0.5)
        self.scorer_err_weight = getattr(args, 'scorer_err_weight', 1.0)    # ModeScorer only
        # TrajScorer's soft ranking target (see _normalized_dist)
        self.conf_tau_min = getattr(args, 'conf_tau_min', 0.02)
        self.conf_sharpen = getattr(args, 'conf_sharpen', 1.0)
        self.conf_goal_blend = getattr(args, 'conf_goal_blend', 0.5)

        # Goal predictor (GOAL_HEATMAP_PLAN.md). The getattr defaults are train.py's, so test_sdd.py
        # and visualize_goal.py rebuild the model as trained from a checkpoint's saved args
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
        ):
            if not ok:
                raise ValueError(message)
        # The goal grid's cell centres r, (G*G, 2): no parameters and nothing in the state_dict
        self.register_buffer('goal_grid', goal_grid(self.goal_grid_cells, self.goal_grid_range),
                             persistent=False)

        # Model components
        self.encoder = HDGT_encoder(input_dim, args)
        self.diffusion_decoder = TransformerDiffusionDecoder(args)
        self.refiner = DiffusionRefiner(args)
        if self.goal_predictor_type == 'heatmap':
            self.goal_predictor = GoalPredictorHeatmap(
                args.hidden_dim, args.obs_len * input_dim, self.goal_grid_cells, hidden=self.goal_hidden,
                use_crop=self.goal_map_crop, use_reach=self.goal_reach_prior, crop_cells=self.goal_crop_cells,
            )
        else:
            self.goal_predictor = GoalPredictorMTP(
                args.hidden_dim, num_modes=self.num_samples,
                with_conf_head=not self.use_scorer,
            )
        self.mode_scorer = (
            ModeScorer(args.hidden_dim, self.pred_len, dropout=scorer_dropout)
            if self.use_scorer and self.scorer_type == 'mode' else None
        )
        self.traj_scorer = (
            TrajScorer(args.hidden_dim, self.pred_len, dropout=scorer_dropout)
            if self.use_scorer and self.scorer_type == 'traj' else None
        )
        # The class statistics behind the s_n floors: no parameters and nothing in the state_dict
        # (train.py saves them in the checkpoint's extra dict; test_sdd.py loads them back)
        self.goal_class_scale = GoalClassScale(6)
        self.last_goal_stats = None

        # Controlled by training loop each epoch (start at 0 → ramp up)
        self.goal_curriculum_alpha = 0.0  # 0 = full oracle, 1 = full predicted goal

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

    def _normalized_dist(self, d):
        """Per-agent distances (N, K) -> 0 for the best chain, 1 at the median one: the
        scale of TrajScorer's soft target no longer depends on how far off the agent is"""
        d_min = d.min(dim=1, keepdim=True).values
        d_med = d.median(dim=1, keepdim=True).values
        tau = (d_med - d_min).clamp_min(self.conf_tau_min)
        return (d - d_min) / tau

    def forward(self, input_dic):
        """
        Forward pass for training and inference.
        
        Training returns: (total_loss, diffusion_loss, refiner_loss, type_masks, types)
        Inference returns: (trajectories, type_masks, types)
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
        
        # Create type masks for evaluation
        pred_indice_bool_type_lis = [
            target_agent_types == agent_type_idx 
            for agent_type_idx in range(6)
        ]

        if self.training:
            # ═════════════════════════════════════════════════════════════
            # TRAINING MODE: 3 LOSSES (Goal Regressor + Diffusion + Refiner)
            # ═════════════════════════════════════════════════════════════
            labels_vel = input_dic["label_lis"]      # velocities for diffusion
            labels_pos = input_dic["label_pos_lis"]  # positions for refiner
            gt_goal = input_dic["goal_pos_lis"]      # GT goal endpoint (regressor target)

            # ─────────────────────────────────────────────────────────────
            # LOSS 3: Goal predictor (the goal distribution; --goal_predictor mtp: K WTA heads)
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

            goal_logits = None
            if self.goal_predictor_type == 'heatmap':
                # Every agent trains every cell: cross-entropy against a Gaussian around its true
                # goal. The K goals are then picked from the distribution, without gradient; the
                # decoder and the refiner are conditioned on them as on the heads before.
                cv_end = cv_endpoint(obs_trajectories, self.pred_len)
                heat_logits = self._goal_logits(goal_features, target_agent_ids, target_agent_types,
                                                input_dic, cv_end, s_n).float()
                heat_target = goal_soft_target((gt_goal32 - cv_end.float()) / s_n.float().unsqueeze(-1),
                                               self.goal_grid, self.goal_heatmap_sigma, self.goal_grid_range)
                goal_assign_loss = -(heat_target * F.log_softmax(heat_logits, dim=1)).sum(dim=1).mean()
                predicted_goals, _, _ = self._pick(heat_logits, cv_end, s_n)
            else:
                predicted_goals, goal_logits = self.goal_predictor(goal_features)
                # predicted_goals: (N, K, 2), goal_logits: (N, K) or None

            # Per-goal squared distance to the GT endpoint, per coordinate (scaled^2 units)
            goals32 = predicted_goals.float()
            goal_dists = ((goals32 - gt_goal32.unsqueeze(1)) ** 2).mean(dim=-1)  # (N, K)
            winner_idx = goal_dists.argmin(dim=1)  # (N,)
            num_heads = goal_dists.shape[1]
            if self.goal_predictor_type == 'mtp':
                # hdgt_goal's loss exactly. Relaxed WTA: while the curriculum is still ramping
                # (alpha < 1), spread the regression loss over the top-3 closest heads; afterwards,
                # pure top-1 winner-takes-all
                if self.goal_curriculum_alpha < 1.0:
                    goal_assign_loss = goal_dists.topk(min(3, num_heads), dim=1, largest=False).values.mean()
                else:
                    goal_assign_loss = goal_dists.gather(1, winner_idx.unsqueeze(1)).mean()
            goal_loss = goal_assign_loss

            # Confidence head (mtp without a scorer, as in hdgt_goal_WTA_regressor): learn the winner
            if goal_logits is not None:
                goal_cls_loss = F.cross_entropy(goal_logits, winner_idx)
                goal_loss = goal_loss + 0.5 * goal_cls_loss

            # Stashed for the training log (train.py)
            self.last_goal_stats = {"assign": goal_assign_loss.detach()}

            # Only now, after this batch's loss, do its agents enter the class scales: no agent's
            # own GT sets its floor
            self.goal_class_scale.update(gt_goal32, target_agent_types)

            # Logged separately: best-goal MSE — the interpretable goal quality
            # (pixel RMSE = sqrt(value) * 50). Occupies the old goal_recon slot.
            goal_recon_loss = goal_dists.gather(1, winner_idx.unsqueeze(1)).mean().detach()

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
            #     is conditioned on head k. At alpha=0 every chain sees GT (old
            #     behavior); at alpha=1 each chain sees its own predicted mode,
            #     exactly matching inference conditions.
            effective_goals_k = (
                self.goal_curriculum_alpha * predicted_goals.detach() +
                (1.0 - self.goal_curriculum_alpha) * gt_goal.unsqueeze(1)
            )
            # --goal_cond_noise: the denoiser and the refiner are conditioned on these goals plus
            # Gaussian noise of RMS goal_cond_noise * s_n, so in training they are as unreliable as
            # the predicted goals are at test and the decoder learns to trust them only that much.
            # The denoiser's endpoint penalty keeps the clean goal as its target; inference is
            # unchanged. Off (0): the clean goals, as before
            cond_goal, cond_goals_k = effective_goal, effective_goals_k
            if self.goal_cond_noise > 0:
                noise_scale = (self.goal_cond_noise / math.sqrt(2.0)) * s_n   # (N,), per coordinate
                cond_goal = effective_goal + noise_scale[:, None] * torch.randn_like(effective_goal)
                cond_goals_k = effective_goals_k + noise_scale[:, None, None] * torch.randn_like(effective_goals_k)

            # ─────────────────────────────────────────────────────────────
            # LOSS 1: Noise prediction + alpha_t-weighted goal penalty
            # ─────────────────────────────────────────────────────────────
            predicted_noise, actual_noise, goal_penalty = self.diffusion_decoder.training_forward(
                target_agent_features,
                target_agent_ids,
                target_agent_types,
                labels_vel,
                cond_goal,
                input_dic,
                goal_target=effective_goal if self.goal_cond_noise > 0 else None
            )
            diffusion_loss = F.mse_loss(predicted_noise, actual_noise) + goal_penalty

            # ─────────────────────────────────────────────────────────────
            # LOSS 2: Multi-Step Refiner
            # ─────────────────────────────────────────────────────────────
            refiner_loss = torch.tensor(0.0, device=predicted_noise.device)
            scorer_term = torch.tensor(0.0, device=predicted_noise.device)

            if self.use_refiner:
                with torch.no_grad():
                    diffusion_samples = self.diffusion_decoder.ddim_sample(
                        target_agent_features,
                        target_agent_ids,
                        target_agent_types,
                        cond_goals_k,   # (N, K, 2): chain k conditioned on head k
                        input_dic,
                        num_samples=self.num_samples,
                        ddim_steps=20
                    )

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
                # goal. The goal still conditions the refiner as an input; we just don't
                # add a separate penalty that would fight the GT supervision.
                refiner_loss = best_mode_distances.mean()

                # ─────────────────────────────────────────────────────────
                # LOSS 4: scorer — rank the chains we actually select from
                # ─────────────────────────────────────────────────────────
                if self.use_scorer:
                    final_traj = all_refined_steps[:, -1].detach()   # (N, K, T, 2)

                    if self.scorer_type == 'mode':
                        # Per-chain ADE, the SAME definition the report ranks on.
                        # (refiner_loss above uses RMSE-over-time; its argmin is a
                        # slightly different chain, so do not reuse best_mode_idx.)
                        ade_k = (final_traj - labels_pos.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
                        ade_winner = ade_k.argmin(dim=1)                 # (N,)

                        scorer_logits, scorer_log_err = self.mode_scorer(
                            target_agent_features, cond_goals_k,   # the goals these chains were drawn with
                            final_traj, obs_trajectories,
                        )
                        scorer_cls_loss = F.cross_entropy(scorer_logits, ade_winner)
                        scorer_err_loss = F.smooth_l1_loss(
                            scorer_log_err, torch.log(ade_k + 1e-3)
                        )
                        scorer_loss = scorer_cls_loss + self.scorer_err_weight * scorer_err_loss
                        self.last_scorer_err = scorer_err_loss.detach()
                    else:
                        # TrajScorer: soft target softmax(-z), z = each chain's path
                        # error plus conf_goal_blend x its goal error, both scaled
                        # per agent (see _normalized_dist)
                        v0 = (obs_trajectories[:, -1] - obs_trajectories[:, -2]).detach()
                        with torch.no_grad():
                            ctx, ctx_mask = self.diffusion_decoder.get_context_features(
                                target_agent_ids, input_dic)
                        scorer_logits = self.traj_scorer(
                            final_traj, v0, target_agent_features.detach(), ctx, ctx_mask)
                        goal_eucl = (predicted_goals.detach() - gt_goal.unsqueeze(1)).norm(dim=-1)
                        d_traj = distances[:, -1, :].detach()
                        z = self.conf_sharpen * (self._normalized_dist(d_traj)
                                                 + self.conf_goal_blend * self._normalized_dist(goal_eucl))
                        soft_target = F.softmax(-z, dim=1)
                        scorer_cls_loss = F.kl_div(F.log_softmax(scorer_logits, dim=1), soft_target,
                                                   reduction='batchmean')
                        scorer_loss = scorer_cls_loss
                        self.last_scorer_err = None

                    # Before the curriculum completes, every chain is conditioned
                    # on a goal blended toward GT, so the ranking target is noise.
                    # Gate with a float rather than skipping the term, so the
                    # scorer's parameters still receive (zero) gradient and DDP
                    # does not need find_unused_parameters.
                    gate = 1.0 if self.goal_curriculum_alpha >= 1.0 else 0.0
                    scorer_term = gate * self.scorer_loss_weight * scorer_loss
                    # Stashed UNGATED so the log shows the scorer's true state
                    # even while it is still gated off.
                    self.last_scorer_loss = scorer_loss.detach()
                    self.last_scorer_cls = scorer_cls_loss.detach()

            # ─────────────────────────────────────────────────────────────
            # TOTAL LOSS
            # ─────────────────────────────────────────────────────────────
            # scorer_term is kept out of goal_loss on purpose: every scorer
            # input is detached, so this term can only move the scorer's weights.
            # Nothing it does can degrade the goal predictor or the generator.
            total_loss = (
                self.diffusion_loss_weight * diffusion_loss +
                self.refiner_loss_weight * refiner_loss +
                self.goal_loss_weight * goal_loss +
                scorer_term
            )

            return total_loss, diffusion_loss, refiner_loss, goal_loss, goal_recon_loss, pred_indice_bool_type_lis, target_agent_types

        else:
            # ═════════════════════════════════════════════════════════════
            # INFERENCE MODE: K multimodal goals, one per DDIM chain
            # ═════════════════════════════════════════════════════════════
            # Each of the K chains is conditioned on its own goal mode, matching
            # the per-chain training conditioning at curriculum alpha=1.
            # Diversity comes from K distinct goals AND K DDIM noise seeds.
            obs_trajectories = all_agent_raw_traj[target_agent_ids]
            self.last_target_obs = obs_trajectories.detach()   # observed positions, for s_n
            self.last_goal_prob = None
            if self.goal_predictor_type == 'heatmap':
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
                predicted_goals, goal_logits = self.goal_predictor(
                    target_agent_features.detach()
                )  # (batch, K, 2), (batch, K)
            # The goals in slot order (slot k = chain k), before any scorer permutes the chains,
            # with the confidence logits: the GOAL HEAD diagnostics read these
            self.last_goals_head_order = predicted_goals.detach()
            self.last_goal_logits = goal_logits.detach() if goal_logits is not None else None

            trajectories = self.diffusion_decoder.ddim_sample(
                target_agent_features,
                target_agent_ids,
                target_agent_types,
                predicted_goals,
                input_dic,
                num_samples=self.num_samples,
                ddim_steps=20
            )

            if self.use_refiner:
                trajectories, _ = self.refiner(
                    trajectories,
                    obs_trajectories,
                    target_agent_features,
                    predicted_goals,
                    return_all_steps=False
                )

            # Rank the K chains so index 0 is the one we predict is best. This
            # is a pure output permutation AFTER generation (the head-k <->
            # modal-emb-k pairing seen during training is untouched):
            # minADE/minFDE are unaffected, while single-mode ADE/FDE
            # (preds[:, 0]) come from the ranking. Without a scorer the chains
            # keep the goal heads' order, as in hdgt_goal_WTA_regressor.
            self.last_pred_error = None
            self.last_mode_probs = None
            if self.use_scorer:
                if self.scorer_type == 'mode':
                    # Scored from the realised chains — same inputs as at training.
                    scorer_logits, scorer_log_err = self.mode_scorer(
                        target_agent_features, predicted_goals,
                        trajectories, obs_trajectories,
                    )
                    conf_order = torch.argsort(scorer_logits, dim=1, descending=True)
                    # Calibrated absolute error estimate, in scaled units, in the
                    # same order as the returned chains (BRIDGE Stage D needs this;
                    # a within-agent softmax cannot provide it).
                    self.last_pred_error = torch.exp(
                        scorer_log_err.gather(1, conf_order)
                    ).detach()
                else:
                    v0 = obs_trajectories[:, -1] - obs_trajectories[:, -2]
                    ctx, ctx_mask = self.diffusion_decoder.get_context_features(
                        target_agent_ids, input_dic)
                    scorer_logits = self.traj_scorer(
                        trajectories, v0, target_agent_features.detach(), ctx, ctx_mask)
                    conf_order = torch.argsort(scorer_logits, dim=1, descending=True)
                # The scorer's probability for each chain, in the returned order
                self.last_mode_probs = torch.softmax(scorer_logits, dim=1).gather(
                    1, conf_order).detach()
                trajectories = trajectories.gather(
                    1,
                    conf_order[:, :, None, None].expand(
                        -1, -1, trajectories.shape[2], trajectories.shape[3]
                    )
                )
                predicted_goals = predicted_goals.gather(
                    1, conf_order[:, :, None].expand(-1, -1, 2))

            # The goals in the same order as the returned chains (goal k belongs
            # to chain k) for downstream consumers (visualization); attribute
            # stash keeps the forward() return signature unchanged.
            self.last_predicted_goals = predicted_goals.detach()

            self.output_is_positions = self.use_refiner

            return trajectories, pred_indice_bool_type_lis, target_agent_types


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
