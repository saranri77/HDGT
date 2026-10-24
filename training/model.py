"""
model.py  (hdgt_map_gnn)
========

hdgt_goal plus the scene map, which reaches the model THROUGH THE GRAPH ONLY.
Grass and obstacle shapes (sdd/shapes, drawn with draw_shapes.py) are "map"
nodes; map->map edges link nearby shapes and map->agent edges link each agent to
the shapes near it. In every GNN layer each agent attends to its agent
neighbours and, separately, to its nearby shapes (AgentHetGNN), and the shapes
update over each other (MapHetGNN). The map thereby shapes the 256-D agent
embedding that the goal predictor, the diffusion decoder and the refiner read;
nothing else sees the map. Those three modules, the scorers and every loss are
hdgt_goal's, unchanged.

Optional map penalties, each off by default and weighted in [0, 1]
(--goal_map_penalty, --diffusion_map_penalty, --refiner_map_penalty): a point of
a predicted goal or path inside a shape its agent may not enter (obstacles for
every agent, grass for every agent but pedestrians) costs its depth inside,
linearly (see map_depth); a point outside costs nothing.

Self-contained copy of the full HDGT model in which the goal predictor is a
**multimodal K-head Winner-Takes-All (MTP-style) predictor** `GoalPredictorMTP`:

    agent_features (N, hidden) -> K candidate goals (N, K, 2) + confidences (N, K)

with K = args.num_prediction (= number of DDIM chains), so each diffusion chain
is conditioned on its OWN goal mode at inference. Multimodality comes from the
loss structure (winner-takes-all: only the head closest to GT gets regression
gradient, so heads specialize into distinct modes), NOT from a latent — there is
no KL and posterior collapse is structurally impossible.

Loss = WTA regression (relaxed top-3 during the curriculum ramp, annealing to
top-1); without a scorer, + 0.5 * cross-entropy on the winner index (trains the
confidence head).

Ranking the K refined chains (index 0 is what single-trajectory metrics read):
  - --scorer_type mode: ModeScorer, hand-made geometric features of each chain
  - --scorer_type traj: TrajScorer, the raw chains with attention across them
    and to the neighbours
  - --use_scorer False: no scorer; the model is just goal-aware and behaves
    like hdgt_goal_WTA_regressor (confidence head trained, chains left in the
    goal heads' order)
The scorer's loss (weight --scorer_loss_weight) starts once the goal curriculum
is done. Every scorer input is detached, so it can only move the scorer.

Training conditioning:
  - diffusion training_forward: winner goal blended with GT via the curriculum
    (the GT trajectory's endpoint ~= winner goal, so this is consistent).
  - ddim_sample + refiner: per-chain goals (N, K, 2), each blended with GT via
    the curriculum — at alpha=1 this exactly matches inference.

This file has NO dependency on model_goal.py; it is meant to be deployed alone
alongside train.py and sdd_dataset.py. The main class is `HDGT_model`
so the train/test scripts (which call model_module.HDGT_model) work unchanged.
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
# Map elements: the graph's "map" nodes and the optional penalties
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
# Agent-only GNN (Simplified from original - no lane/polygon)
# =============================================================================

class AgentHetGNN(nn.Module):
    """One layer of agent updates.

    Each agent fuses three streams: itself (self-loop edge), attention over its
    agent neighbours ("other" edges), and with use_map, attention over the map
    elements near it ("m2a" edges). Queries are per agent type; keys/values per
    SOURCE type (agent type for "other", map class for "m2a"). Edge states are
    refreshed from [source, edge, destination] like the original HDGT.
    With use_map=False the module and its parameter names are exactly the
    agent-only version, so old checkpoints still load.
    """
    def __init__(self, args):
        super().__init__()
        self.norm = nn.LayerNorm
        self.hidden_dim = args.hidden_dim
        self.head_dim = args.head_dim
        self.d_model = args.hidden_dim
        self.n_head = self.d_model//args.head_dim
        self.use_map = bool(getattr(args, "use_map", False))
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
    """One layer of map-node updates (use_map only).

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
    """HDGT encoder: agent nodes, plus map nodes when args.use_map."""
    def __init__(self, input_dim, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.use_map = bool(getattr(args, "use_map", False))
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
    winner-takes-all loss (computed in HDGT_model.forward), which lets each of
    the K heads specialize on a distinct future mode instead of averaging them.
    No latent, no KL — nothing can collapse.

    `with_conf_head` controls the legacy blind confidence head. When the
    ModeScorer is in use it is built with with_conf_head=False, so the trunk
    that produces the goals receives gradient ONLY from the WTA regression.
    That matters: in scaled units the WTA MSE is ~1e-2 while a 20-way
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
                         map_geom=None, map_mask=None):
        """Training forward pass with noise prediction and alpha_t-weighted goal loss.

        The model predicts noise (epsilon). For the goal penalty, we recover
        x0 (velocities) from the noise prediction, convert to positions via
        cumsum, and compare the final position to the goal. The penalty is
        weighted by alpha_t so it only applies at low noise levels where x0
        recovery is accurate.

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
        goal_penalty = torch.mean(alpha_t * (final_pos - goal_pos) ** 2)

        if map_geom is not None:
            depth = map_depth(densify_path(predicted_positions), map_geom, map_mask)   # (batch, 2T)
            map_penalty = torch.mean(alpha_t.squeeze(-1) * depth.mean(dim=-1))
        else:
            map_penalty = torch.zeros((), device=device)

        return predicted_noise, noise, goal_penalty, map_penalty

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
        3. Goal Loss: K-head WTA regression (+ confidence cross-entropy without a scorer)
        4. Scorer Loss: ranks the K refined chains (ModeScorer or TrajScorer), weight
           --scorer_loss_weight, once the goal curriculum is done
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
        # --use_scorer False the model is just goal-aware, as in
        # hdgt_goal_WTA_regressor: the goal predictor keeps its confidence head
        # and the chains stay in the goal heads' order.
        self.use_scorer = flag_on(getattr(args, 'use_scorer', True)) and self.use_refiner
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

        # Map penalties, each optional and off by default: the map reaches the
        # model through the graph only (HDGT_encoder). Switched on, each point of
        # a goal or path inside a shape its agent may not enter costs its depth
        # there (map_depth: linear, 0 outside), times a weight in [0, 1]. With
        # --map_penalty_gt_exempt (default), shapes the agent's own GT path enters
        # are left out for that agent, so a penalty never pushes against the GT.
        self.goal_map_penalty = flag_on(getattr(args, 'goal_map_penalty', False))
        self.diffusion_map_penalty = flag_on(getattr(args, 'diffusion_map_penalty', False))
        self.refiner_map_penalty = flag_on(getattr(args, 'refiner_map_penalty', False))
        self.goal_map_penalty_weight = float(getattr(args, 'goal_map_penalty_weight', 0.1))
        self.diffusion_map_penalty_weight = float(getattr(args, 'diffusion_map_penalty_weight', 0.1))
        self.refiner_map_penalty_weight = float(getattr(args, 'refiner_map_penalty_weight', 0.1))
        for part in ("goal", "diffusion", "refiner"):
            weight = getattr(self, f"{part}_map_penalty_weight")
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"--{part}_map_penalty_weight must be in [0, 1], got {weight}")
        if self.refiner_map_penalty and not self.use_refiner:
            raise ValueError("--refiner_map_penalty needs the refiner (--use_refiner True)")
        self.map_penalty_gt_exempt = flag_on(getattr(args, 'map_penalty_gt_exempt', True))
        self.last_map_penalties = {}     # the switched-on penalties of the last training batch

        # Model components
        self.encoder = HDGT_encoder(input_dim, args)
        self.diffusion_decoder = TransformerDiffusionDecoder(args)
        self.refiner = DiffusionRefiner(args)
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

        # Controlled by training loop each epoch (start at 0 → ramp up)
        self.goal_curriculum_alpha = 0.0  # 0 = full oracle, 1 = full predicted goal

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

            # Map penalties (each off by default): the shapes each target may not
            # enter, in its own frame, for the switched-on penalties only
            zero = torch.zeros((), device=labels_pos.device)
            goal_map_loss = diffusion_map_loss = refiner_map_loss = zero
            map_geom = map_mask = None
            if self.goal_map_penalty or self.diffusion_map_penalty or self.refiner_map_penalty:
                if "map_geom_lis" not in input_dic:
                    raise ValueError("a map penalty is on but the batch has no map: it needs --use_map True")
                map_geom = input_dic["map_geom_lis"]
                map_mask = input_dic["map_loss_mask_lis" if self.map_penalty_gt_exempt
                                     else "map_forbid_mask_lis"]

            # ─────────────────────────────────────────────────────────────
            # LOSS 3: Multimodal Goal Predictor (K-head Winner-Takes-All)
            # ─────────────────────────────────────────────────────────────
            # Detach encoder features so goal gradients don't flow into the
            # encoder. The goal predictor is a projection head that reads
            # encoder outputs — the encoder is trained by diffusion + refiner.
            predicted_goals, goal_logits = self.goal_predictor(target_agent_features.detach())
            # predicted_goals: (N, K, 2), goal_logits: (N, K)

            # Per-head squared distance to the GT endpoint
            goal_dists = ((predicted_goals - gt_goal.unsqueeze(1)) ** 2).mean(dim=-1)  # (N, K)
            winner_idx = goal_dists.argmin(dim=1)  # (N,)

            # Relaxed WTA: while the curriculum is still ramping (alpha < 1),
            # spread the regression loss over the top-3 closest heads so no head
            # dies before it ever wins; afterwards, pure top-1 winner-takes-all.
            if self.goal_curriculum_alpha < 1.0:
                topk = min(3, goal_dists.shape[1])
                goal_wta_loss = goal_dists.topk(topk, dim=1, largest=False).values.mean()
            else:
                goal_wta_loss = goal_dists.gather(1, winner_idx.unsqueeze(1)).mean()

            # Confidence head: learn which GOAL is closest to the GT goal. Only
            # there without a scorer (as in hdgt_goal_WTA_regressor) — see
            # ModeScorer for why this target is the wrong one for ranking.
            if goal_logits is not None:
                goal_cls_loss = F.cross_entropy(goal_logits, winner_idx)
                goal_loss = goal_wta_loss + 0.5 * goal_cls_loss
            else:
                goal_loss = goal_wta_loss
            # Logged separately: best-head MSE — the interpretable goal quality
            # (pixel RMSE = sqrt(value) * 50). Occupies the old goal_recon slot.
            goal_recon_loss = goal_dists.gather(1, winner_idx.unsqueeze(1)).mean().detach()

            if self.goal_map_penalty:     # every head's goal point, inside a forbidden shape
                goal_map_loss = map_depth(predicted_goals, map_geom, map_mask).mean()

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

            # ─────────────────────────────────────────────────────────────
            # LOSS 1: Noise prediction + alpha_t-weighted goal penalty
            # ─────────────────────────────────────────────────────────────
            predicted_noise, actual_noise, goal_penalty, diffusion_map_loss = self.diffusion_decoder.training_forward(
                target_agent_features,
                target_agent_ids,
                target_agent_types,
                labels_vel,
                effective_goal,
                input_dic,
                map_geom=map_geom if self.diffusion_map_penalty else None,
                map_mask=map_mask,
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
                        effective_goals_k,   # (N, K, 2): chain k conditioned on head k
                        input_dic,
                        num_samples=self.num_samples,
                        ddim_steps=20
                    )

                obs_trajectories = all_agent_raw_traj[target_agent_ids]

                all_refined_steps, _ = self.refiner(
                    diffusion_samples,
                    obs_trajectories,
                    target_agent_features,
                    effective_goals_k,   # (N, K, 2): per-chain goal conditioning
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

                if self.refiner_map_penalty:   # every refined chain, densified
                    refiner_map_loss = map_depth(
                        densify_path(all_refined_steps[:, -1]), map_geom, map_mask).mean()

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
                            target_agent_features, effective_goals_k,
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
            # Switched-off map penalties are 0 here, so they add nothing
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
                scorer_term +
                map_term
            )

            return total_loss, diffusion_loss, refiner_loss, goal_loss, goal_recon_loss, pred_indice_bool_type_lis, target_agent_types

        else:
            # ═════════════════════════════════════════════════════════════
            # INFERENCE MODE: K multimodal goals, one per DDIM chain
            # ═════════════════════════════════════════════════════════════
            # Each of the K chains is conditioned on its own goal mode, matching
            # the per-chain training conditioning at curriculum alpha=1.
            # Diversity comes from K distinct goals AND K DDIM noise seeds.
            predicted_goals, goal_logits = self.goal_predictor(
                target_agent_features.detach()
            )  # (batch, K, 2), (batch, K)

            trajectories = self.diffusion_decoder.ddim_sample(
                target_agent_features,
                target_agent_ids,
                target_agent_types,
                predicted_goals,
                input_dic,
                num_samples=self.num_samples,
                ddim_steps=20
            )

            obs_trajectories = all_agent_raw_traj[target_agent_ids]

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
