"""
model.py
========

Self-contained copy of the full HDGT model in which the goal predictor is a
**multimodal K-head Winner-Takes-All (MTP-style) predictor** `GoalPredictorMTP`:

    agent_features (N, hidden) -> K candidate goals (N, K, 2)

with K = args.num_prediction (= number of DDIM chains), so each diffusion chain
is conditioned on its OWN goal mode at inference. Multimodality comes from the
loss structure (winner-takes-all: only the head closest to GT gets regression
gradient, so heads specialize into distinct modes), NOT from a latent — there is
no KL and posterior collapse is structurally impossible.

Loss = WTA regression (relaxed top-3 during the curriculum ramp, annealing to
top-1).

The generator (encoder, goal predictor, diffusion decoder, refiner) is the one
of hdgt_goal. What this version adds is the ranker, TrajRankNN, which orders
the K refined chains (index 0 is what single-trajectory metrics read) in place
of hdgt_goal's scorers. It is part of HDGT_model and trains ONLINE with it, on
the same chains the refiner loss is computed on, to order them by their ADE to
the ground truth. Its loss (weight --ranker_loss_weight) starts once the goal
curriculum is done. Every ranker input is detached, so it can only move the
ranker: minADE / minFDE are the generator's alone.

Map (--use_map, off by default), exactly as hdgt_map_gnn: the scene map reaches
the model THROUGH THE GRAPH ONLY. Grass and obstacle shapes (sdd/shapes, drawn
with draw_shapes.py) are "map" nodes; map->map edges link nearby shapes and
map->agent edges link each agent to the shapes near it. In every GNN layer each
agent attends to its agent neighbours and, separately, to its nearby shapes
(AgentHetGNN), and the shapes update over each other (MapHetGNN). The map
thereby shapes the agent embedding that the goal predictor, the diffusion
decoder, the refiner and the ranker read; nothing else sees the map. With
--use_map False every module, parameter and loss is the map-free model's.

Optional map penalties, each off by default, weighted in [0, 1] and needing
--use_map (--goal_map_penalty, --diffusion_map_penalty, --refiner_map_penalty):
a point of a predicted goal or path inside a shape its agent may not enter
(obstacles for every agent, grass for every agent but pedestrians) costs its
depth inside, linearly (see map_depth); a point outside costs nothing.

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


def _within_std(x, dim):
    """Standardise each channel across `dim` (the K axis), per agent.

    Makes a feature answer "how does this candidate compare to its siblings",
    which is scale-free and is the question a ranker is actually asked. An agent
    whose candidates are all identical gives 0 rather than NaN.
    """
    m = x.mean(dim=dim, keepdim=True)
    s = x.std(dim=dim, keepdim=True).clamp_min(1e-4)
    return (x - m) / s


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
    """

    def __init__(self, hidden_dim, pred_len, K, n_state=8, n_geom=28,
                 d=64, n_heads=4, n_mode_layers=2, n_types=6, dropout=0.2,
                 within_norm=False, norm_momentum=0.01):
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
        # within_norm doubles the input width: each channel enters twice, once
        # globally normalised and once standardised across the agent's K.
        #
        # DEFAULT OFF because it was MEASURED NOT TO HELP on this data. The idea
        # was that global stats would crush within-agent variation, but on real
        # candidates the median within/global std ratio is 0.711, not the ~0.05
        # the hypothesis predicted, so the features were never starved. The A/B
        # at 8 epochs: off closes 62.8% of the gap, on closes 61.9%. Kept as an
        # option for ablation, not because the feature earns its place.
        self.within_norm = within_norm
        mult = 2 if within_norm else 1
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
        self.state_proj = nn.Linear(n_state * mult, d)
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)

        # stage one, part B: the per-mode scalars
        self.geom_mlp = nn.Sequential(
            nn.Linear(n_geom * mult, d), nn.ReLU(inplace=True),
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

    def forward(self, state, geom, agent_types, ctx_tokens=None, ctx_mask=None):
        """
        state       (N, K, T, n_state)
        geom        (N, K, n_geom)  -- includes the 4 per-agent kinematic scalars
                    that replaced the encoder feature (see build_rank_features)
        agent_types (N,)  ZERO-BASED, 0..5, matching target_agent_types inside
                    HDGT_model (AGENT_TYPE_NAMES is 1..6). Getting the base
                    wrong collapses pedestrian onto biker and raises nothing.
        ctx_tokens  (N, M, hidden_dim) or None
        ctx_mask    (N, M) bool, True = padding, or None
        """
        n, k, t, _ = state.shape
        assert k <= self.slot_emb.shape[0], (
            f"ranker was trained for K={self.slot_emb.shape[0]}, got K={k}")
        # Two views of every channel, concatenated.
        #
        # GLOBAL: (x - mean) / std over the training batches. Carries absolute
        # scale, which the log_ade head needs to say how wrong a chain is in pixels.
        #
        # WITHIN-AGENT: standardised across the K axis, per agent. This is the
        # one that matters for ranking, and it was missing. Global stats are
        # dominated by differences BETWEEN agents — a parked car's endpoint is
        # ~0.05 scaled units, a biker's ~6 — so the differences WITHIN one
        # agent's 20 candidates shrink to a few percent of the normalised scale
        # (measured on this data: 0.034 for cars, 0.109 for pedestrians). The
        # first linear layer then sees a large per-agent offset with a tiny
        # perturbation on top, and the only input still varying across K at full
        # scale is the slot embedding — which is precisely the best_fixed_slot
        # rule the network was struggling to beat. Ranking is a within-agent
        # question, so it gets a within-agent normalisation.
        if self.training:
            self._track_norm_stats(state, geom)
        if self.within_norm:
            state = torch.cat([(state - self.state_mean) / self.state_std,
                               _within_std(state, dim=1)], dim=-1)
            geom = torch.cat([(geom - self.geom_mean) / self.geom_std,
                              _within_std(geom, dim=1)], dim=-1)
        else:
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


# ===========================================================================
# RankNN loss: soft-pick regret + soft-rank Spearman (+ optional listwise)
#              + auxiliary magnitude
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


def topk_listwise_loss(err_score, ade_k, k_top=3, tau=5.0):
    """Truncated Plackett-Luce over the first k_top positions.

    Spearman weights every pair equally, but getting rank 19 right against rank
    20 is worth nothing for top-1 selection. This concentrates the gradient
    where the reported metric actually reads.

    tau is in PIXELS, matching ade_k. This matters: the reference spec set
    tau=0.1, which is a scaled-unit value. At pixel scale the soft target
    collapses to one-hot (effective support 1.00 of 20), which would silently
    restore the hard-argmin-on-seed-noise target this loss exists to avoid. At
    tau=5 px the support is about 6 of 20, which is genuinely soft.
    """
    target = F.softmax(-ade_k / tau, dim=-1)
    scores = -err_score                            # high = good, for log_softmax
    loss = torch.zeros(err_score.shape[0], device=err_score.device)
    masked = scores
    avail = torch.ones_like(scores, dtype=torch.bool)
    for _ in range(min(k_top, scores.shape[-1])):
        logp = torch.log_softmax(masked.masked_fill(~avail, -1e9), dim=-1)
        loss = loss - (target * avail * logp).sum(-1)
        # remove the truly-best remaining candidate and repeat
        nxt = (ade_k + (~avail) * 1e9).argmin(dim=-1)
        avail = avail.clone()
        avail.scatter_(1, nxt.unsqueeze(1), False)
    return loss


def soft_pick_loss(err_score, ade_k, tau=1.0):
    """Expected REGRET of a soft top-1 pick, in pixels.

        p_k  = softmax(-z_k / tau),  z = err_score standardised within the agent
        loss = sum_k p_k * ade_k  -  min_k ade_k

    This is the metric the report reads (top-1 ADE), made differentiable. It is
    added because the two existing top-heavy terms optimise proxies of it and
    the proxies disagree with it in exactly the cases that matter:

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
              lam_top=0.0, lam_reg=0.2, reg=0.1, tau=5.0, tau_pick=1.0,
              k_top=3, pick_scale=10.0):
    """Full objective, aggregated per type then across types.

    Term weights (defaults changed with the soft-pick loss):
      lam_pick  soft-pick regret (pixels / pick_scale). The primary term now:
                it IS the reported metric, softened.
      lam_sp    Spearman, kept at a small weight. It gives every candidate a
                gradient even when the soft pick has become nearly hard, and
                keeps the full ordering meaningful, which the bridge plan's
                per-candidate confidence needs and top-1 regret does not.
      lam_top   Plackett-Luce, off by default. It targets the same thing as
                the regret term through a less faithful proxy; left available
                for ablation.
      lam_reg   auxiliary log-ADE regression, unchanged.

    pick_scale converts the regret from pixels (typically 3-30) to O(1) so the
    weights above mean what they say relative to the Spearman term in [-1, 1].

    Type-level aggregation is not cosmetic: pedestrians outnumber bikers by an
    order of magnitude, so a plain mean over the batch tunes the ranker to
    pedestrians alone and the biker rows never move.
    """
    per_agent = (lam_sp * spearman_loss(err_score, ade_k, reg)
                 + lam_pick * soft_pick_loss(err_score, ade_k, tau_pick) / pick_scale
                 + lam_reg * F.smooth_l1_loss(
                     log_ade, torch.log(ade_k + 1e-3), reduction="none").mean(-1))
    if lam_top > 0:
        per_agent = per_agent + lam_top * topk_listwise_loss(err_score, ade_k, k_top, tau)
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
        self.N_SETTLE = 5
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
# Main Model: HDGT with Diffusion + Refiner
# =============================================================================

def flag_on(value):
    """A flag that may arrive as a bool or as argparse's "True"/"False" string"""
    return value in (True, 'True', 'true', '1')


class HDGT_model(nn.Module):
    """
    HDGT model with diffusion decoder, refiner and ranker.

    Architecture:
        1. Encoder: HDGT-style heterogeneous graph network
        2. Diffusion Decoder: Generates diverse K samples (noise prediction)
        3. Refiner: Polishes samples for accuracy (5 iterations)
        4. Ranker: orders the K refined samples, best first (TrajRankNN)

    Training Losses:
        1. Diffusion Loss: noise prediction + alpha_t-weighted goal penalty
        2. Multi-Step Refiner Loss: best-of-K distance to GT (no goal penalty)
        3. Goal Loss: K-head WTA regression
        4. Ranker Loss: orders the K refined chains by their ADE to GT, weight
           --ranker_loss_weight, once the goal curriculum is done
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

        # The ranker orders the K chains. Its loss starts once the goal
        # curriculum is done, as hdgt_goal's scorer's did: before that, every
        # chain is conditioned on a goal blended toward GT, so the order to
        # learn is not the one inference produces.
        self.ranker_loss_weight = getattr(args, 'ranker_loss_weight', 0.5)

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
        self.goal_predictor = GoalPredictorMTP(args.hidden_dim, num_modes=self.num_samples)
        # Every ranker input is detached, so its loss cannot reward the generator
        # for chains that are easy to order (the cheapest way to get those is
        # fewer distinct modes, which minADE would pay for), and train.py gives
        # it its own parameter group and gradient clip.
        self.ranker = TrajRankNN(
            args.hidden_dim, self.pred_len, self.num_samples,
            n_geom=19 + self.diffusion_decoder.N_SETTLE + 4,   # see build_rank_features
        )

        # Controlled by training loop each epoch (start at 0 → ramp up)
        self.goal_curriculum_alpha = 0.0  # 0 = full oracle, 1 = full predicted goal
        # Ranker soft-pick temperature, annealed across the run (progress 0 -> 1,
        # set by train.py): soft at first, so every chain gets gradient and the
        # whole order is learnt, close to the hard argmin used at inference by
        # the end. In units of each agent's own score spread.
        self.ranker_progress = 0.0
        self.ranker_tau_pick_start = 2.0
        self.ranker_tau_pick_end = 0.3
        # The ranker's target, per-chain ADE, is in PIXELS, the unit rank_loss's
        # temperatures are written in. train.py passes the dataset's scale.
        self.ranker_pixel_scale = float(getattr(args, 'scale_factor', 50.0))

    def _rank_chains(self, diffusion_samples, all_refined_steps, goals, obs_trajectories,
                     target_agent_ids, target_agent_types, input_dic):
        """Score the K chains with the ranker. Returns (positions, err_score, log_ade):

            positions  (N, K, T, 2)  the chains it scored, as POSITIONS relative
                                     to the last observation: the refined ones,
                                     or the decoder's velocities summed when the
                                     refiner is off
            err_score  (N, K)        LOW is good, so sort ascending
            log_ade    (N, K)        predicted log ADE of each chain, in pixels

        All in slot order. The one way into the ranker, for training and
        inference alike, so it is fit on exactly the features it ranks with.
        Its inputs are built under no_grad: the ranker learns from the
        generator's output and sends no gradient back into it. The DDIM
        settling diagnostic is read off the decoder, where
        ddim_sample(collect_settle=True) leaves it.
        """
        with torch.no_grad():
            if all_refined_steps is None:
                positions = torch.cumsum(diffusion_samples, dim=2)
                # zero is the honest effort for chains no refiner touched
                refine_delta = positions.new_zeros(
                    positions.shape[0], positions.shape[1], self.refiner.refine_num)
            else:
                # all_refined_steps is (N, refine_num+1, K, T, 2)
                positions = all_refined_steps[:, -1]
                # How far the refiner had to drag each chain at each step. A
                # chain needing heavy correction started out bad. (N, K, 5),
                # same axis order as last_settle so the two cannot be mixed up.
                refine_delta = (
                    (all_refined_steps[:, 1:] - all_refined_steps[:, :-1])
                    .norm(dim=-1).mean(dim=-1)      # (N, 5, K)
                    .transpose(1, 2).contiguous()   # (N, K, 5)
                )
            state, geom = build_rank_features(
                positions.float(), goals.float(), obs_trajectories.float(),
                refine_delta.float(), self.diffusion_decoder.last_settle.float(),
            )
            # Neighbour context tokens, for judging social conflict.
            # Slot-independent: one set per agent, shared by its K.
            ctx, ctx_mask = self.diffusion_decoder.get_context_features(
                target_agent_ids, input_dic)

        err_score, log_ade = self.ranker(state, geom, target_agent_types, ctx, ctx_mask)
        return positions, err_score, log_ade

    def forward(self, input_dic):
        """
        Forward pass for training and inference.
        
        Training returns: (total_loss, diffusion_loss, refiner_loss, type_masks, types)
        Inference returns: (trajectories, type_masks, types), the K trajectories
                           of each agent in the ranker's order, best first
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
            predicted_goals = self.goal_predictor(target_agent_features.detach())
            # predicted_goals: (N, K, 2)

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

            # No confidence-head term: the ranker orders the chains, as in
            # hdgt_goal whenever a scorer does.
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

            # The K chains are sampled whatever the refiner setting, because the
            # ranker learns to order them. With the refiner on this is exactly
            # the call hdgt_goal makes; recording the settling diagnostic (a
            # ranker input) changes no chain.
            with torch.no_grad():
                diffusion_samples = self.diffusion_decoder.ddim_sample(
                    target_agent_features,
                    target_agent_ids,
                    target_agent_types,
                    effective_goals_k,   # (N, K, 2): chain k conditioned on head k
                    input_dic,
                    num_samples=self.num_samples,
                    ddim_steps=20,
                    collect_settle=True,
                )

            obs_trajectories = all_agent_raw_traj[target_agent_ids]
            all_refined_steps = None

            if self.use_refiner:
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

            # ─────────────────────────────────────────────────────────────
            # LOSS 4: ranker — rank the chains we actually select from
            # ─────────────────────────────────────────────────────────────
            # It scores the very chains the refiner loss was just computed on,
            # each beside the goal it was conditioned on (at curriculum alpha=1,
            # exactly the inference conditioning), and learns to order them by
            # per-chain ADE, the SAME definition the report ranks on. These
            # chains were sampled with the denoiser's dropout live, the one
            # difference from inference that training online leaves.
            positions, err_score, log_ade = self._rank_chains(
                diffusion_samples, all_refined_steps, effective_goals_k,
                obs_trajectories, target_agent_ids, target_agent_types, input_dic)
            with torch.no_grad():
                ade_k = ((positions - labels_pos.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
                         * self.ranker_pixel_scale)                    # (N, K), pixels
            tau_pick = self.ranker_tau_pick_start + (
                self.ranker_tau_pick_end - self.ranker_tau_pick_start) * self.ranker_progress
            ranker_loss = rank_loss(err_score.float(), log_ade.float(), ade_k.float(),
                                    target_agent_types, tau_pick=tau_pick)

            # Before the curriculum completes, every chain is conditioned on a
            # goal blended toward GT, so the ranking target is not the one
            # inference poses. Gate with a float rather than skipping the term,
            # so the ranker's parameters still receive (zero) gradient and DDP
            # does not need find_unused_parameters.
            gate = 1.0 if self.goal_curriculum_alpha >= 1.0 else 0.0
            ranker_term = gate * self.ranker_loss_weight * ranker_loss
            # Stashed UNGATED so the log shows the ranker's true state even
            # while it is still gated off.
            self.last_ranker_loss = ranker_loss.detach()

            # ─────────────────────────────────────────────────────────────
            # TOTAL LOSS
            # ─────────────────────────────────────────────────────────────
            # ranker_term is kept out of goal_loss on purpose: every ranker
            # input is detached, so this term can only move the ranker's weights.
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
                ranker_term +
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
            predicted_goals = self.goal_predictor(
                target_agent_features.detach()
            )  # (batch, K, 2)

            diffusion_samples = self.diffusion_decoder.ddim_sample(
                target_agent_features,
                target_agent_ids,
                target_agent_types,
                predicted_goals,
                input_dic,
                num_samples=self.num_samples,
                ddim_steps=20,
                collect_settle=True,   # a ranker input
            )

            obs_trajectories = all_agent_raw_traj[target_agent_ids]

            trajectories = diffusion_samples
            all_refined_steps = None
            if self.use_refiner:
                # return_all_steps hands the ranker the refiner's per-step
                # effort; the final chains are the same either way
                all_refined_steps, _ = self.refiner(
                    diffusion_samples,
                    obs_trajectories,
                    target_agent_features,
                    predicted_goals,
                    return_all_steps=True
                )
                trajectories = all_refined_steps[:, -1]

            # Rank the K chains so index 0 is the one we predict is best. This
            # is a pure output permutation AFTER generation (the head-k <->
            # modal-emb-k pairing seen during training is untouched):
            # minADE/minFDE are unaffected, while single-mode ADE/FDE
            # (preds[:, 0]) come from the ranking.
            _, err_score, log_ade = self._rank_chains(
                diffusion_samples, all_refined_steps, predicted_goals,
                obs_trajectories, target_agent_ids, target_agent_types, input_dic)
            # err_score is LOW-is-good, so ascending: descending would invert
            # the ranker completely and look exactly like a failed training run
            conf_order = torch.argsort(err_score, dim=1, descending=False)
            # The ranker's absolute error estimate for each chain, in the
            # returned order. PIXELS, not scaled units: log_ade is fit against
            # ADE in pixels, so consumers must not multiply by SCALE_FACTOR.
            self.last_pred_error = torch.exp(log_ade.gather(1, conf_order)).detach()
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
