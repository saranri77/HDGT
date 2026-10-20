import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import math
import dgl.function as fn
from functools import partial


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
    def __init__(self, args):
        super().__init__()
        self.norm = nn.LayerNorm
        self.hidden_dim = args.hidden_dim
        self.head_dim = args.head_dim
        self.d_model = args.hidden_dim
        self.n_head = self.d_model//args.head_dim
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

        self.attn_fcs = nn.ModuleList([nn.ModuleDict({
            "other": nn.Sequential(
                nn.Linear(self.n_head * d_v, self.d_model, bias=True),
                nn.ReLU(inplace=True)
            )
        }) for agent_type_index in range(6)])
        
        self.self_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                nn.ReLU(inplace=True)
            ) for _ in range(6)])
        
        self.out_fc = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim, bias=True),
            nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
        ) for _ in range(6)])
                
        self.out_ffn = nn.ModuleList([PositionwiseFeedForward(args.hidden_dim, args.hidden_dim*4, args.dropout, nn.LayerNorm)
            for _ in range(6)])

        self.etype_dic = {
            "other": (partial(self.message_func, etype="other"), partial(self.reduce_func, etype="other"))
        }
        self.etype2hidden_name = {"other": "a_e_hidden"}

        self.edge_MLP = {}
        self.edge_MLP["self"] = nn.ModuleList([MLP(args.hidden_dim, args.hidden_dim, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
            for _ in range(6)])
        self.edge_MLP["other"] = nn.ModuleList([MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
            for _ in range(6)])
        self.edge_MLP = nn.ModuleDict(self.edge_MLP)

    def forward(self, input_dic):
        self.a_e_type_dict = input_dic["a_e_type_dict"]
        self.a_n_type_lis = input_dic["a_n_type_lis"]
        with input_dic["graph_lis"].local_scope():
            self.device = "cuda:"+str(input_dic["gpu"])

            input_q = input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"]
            q = torch.zeros((input_q.shape[0], self.n_head * self.head_dim), device=self.device)
            for agent_type_index in range(6):
                if len(self.a_n_type_lis[agent_type_index]) != 0:
                    q[self.a_n_type_lis[agent_type_index]] = self.wqs[agent_type_index](input_q[self.a_n_type_lis[agent_type_index]])

            other_q = q.view(input_q.shape[0], self.n_head, self.head_dim)
            input_dic["graph_lis"].nodes["agent"].data["other_q"] = other_q

            num_nodes = input_dic["graph_lis"].number_of_nodes("agent")
            input_dic["graph_lis"].nodes["agent"].data["a_n_hidden_out"] = torch.zeros(
                (num_nodes, self.n_head * self.head_dim), device=self.device)

            input_dic["graph_lis"].multi_update_all(etype_dict=self.etype_dic, cross_reducer="stack")
            all_out_fea = input_dic["graph_lis"].nodes["agent"].data["a_n_hidden_out"]

            other_out_tmp_in = all_out_fea.squeeze(1) if len(all_out_fea.shape) == 3 else all_out_fea
            input_self_fea = input_dic["graph_lis"].edges["self"].data["a_e_hidden"]
            all_out_n_fea = torch.zeros_like(input_q)
            for agent_type_index in range(6):
                if len(self.a_n_type_lis[agent_type_index]) != 0:
                    other_out_fea = self.attn_fcs[agent_type_index]["other"](other_out_tmp_in[self.a_n_type_lis[agent_type_index]])
                    self_fea = self.self_fc[agent_type_index](input_self_fea[self.a_n_type_lis[agent_type_index]])
                    out_n_fea = torch.stack([self_fea, other_out_fea], dim=1)
                    out_n_fea = self.out_fc[agent_type_index](out_n_fea.view(out_n_fea.shape[0], -1)) + input_q[self.a_n_type_lis[agent_type_index]]
                    out_n_fea = self.out_ffn[agent_type_index](out_n_fea)
                    all_out_n_fea[self.a_n_type_lis[agent_type_index]] = out_n_fea

            out_e_fea_lis = []
            self_e_fea_tmp_in = input_dic["graph_lis"].edges["self"].data["a_e_hidden"]
            other_e_fea_tmp_in = torch.cat([input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][input_dic["graph_lis"].edges(etype="other")[0]], input_dic["graph_lis"].edges["other"].data["a_e_hidden"], input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"][input_dic["graph_lis"].edges(etype="other")[1]]], dim=-1)
            
            self_e_fea_tmp_out = torch.zeros_like(self_e_fea_tmp_in)
            other_e_fea_tmp_out = torch.zeros((other_e_fea_tmp_in.shape[0], self.hidden_dim), device=self.device)
            for agent_type_index in range(6):
                if len(self.a_e_type_dict["self"][agent_type_index]) != 0:
                    self_e_fea_tmp_out[self.a_e_type_dict["self"][agent_type_index]] = self.edge_MLP["self"][agent_type_index](self_e_fea_tmp_in[self.a_e_type_dict["self"][agent_type_index]])
                if len(self.a_e_type_dict["other"][agent_type_index]) != 0:
                    other_e_fea_tmp_out[self.a_e_type_dict["other"][agent_type_index]] = self.edge_MLP["other"][agent_type_index](other_e_fea_tmp_in[self.a_e_type_dict["other"][agent_type_index]])
            out_e_fea_lis.append(self_e_fea_tmp_out + input_dic["graph_lis"].edges["self"].data["a_e_hidden"])
            out_e_fea_lis.append(other_e_fea_tmp_out + input_dic["graph_lis"].edges["other"].data["a_e_hidden"])
        return all_out_n_fea, out_e_fea_lis

    def message_func(self, edges, etype):
        tmp_input = edges.data[self.etype2hidden_name[etype]]
        tmp_output = torch.zeros((tmp_input.shape[0], self.n_head * self.head_dim * 2), device=self.device)
        for agent_type_index in range(6):
            if len(self.a_e_type_dict["other"][agent_type_index]) != 0:
                tmp_output[self.a_e_type_dict["other"][agent_type_index]] = self.wkvs[etype][agent_type_index](tmp_input[self.a_e_type_dict["other"][agent_type_index]])
        k, v = tmp_output.view(-1, self.n_head, self.head_dim*2).split(dim=-1, split_size=self.head_dim)
        return {etype+"_k":k, etype+"_v":v}

    def reduce_func(self, nodes, etype):
        if nodes.mailbox[etype+"_k"].shape[1] == 0:
            node_num = nodes.mailbox[etype+"_k"].shape[0]
            return {"a_n_hidden_out": torch.zeros((node_num, self.n_head * self.head_dim), device=self.device)}

        node_num, neighbor_num, n_head, hidden = nodes.mailbox[etype+"_k"].shape
        q = nodes.data[etype+"_q"].view(node_num*self.n_head, -1).unsqueeze(1)
        k = nodes.mailbox[etype+"_k"].transpose(1, 2).reshape(node_num*self.n_head, neighbor_num, -1)
        v = nodes.mailbox[etype+"_v"].transpose(1, 2).reshape(node_num*self.n_head, neighbor_num, -1)
        output, attn = self.attention(q, k, v, mask=None)
        return {"a_n_hidden_out":output.view(node_num, -1)}


class HDGT_encoder(nn.Module):
    """HDGT encoder - simplified agent-only version"""
    def __init__(self, input_dim, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.shared_coor_encoder = MLP(d_in=2, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//4, norm=None)
        self.shared_rel_encoder = MLP(d_in=4, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//4, norm=None)
        
        self.agent_emb = Agent2embedding(input_dim, args)
        self.temporal_encoders = torch.nn.ModuleList([AgentTemporalEncoder(args) for _ in range(6)])
        self.agent_e_fea_MLPs = torch.nn.ModuleList([MLP(args.hidden_dim+args.hidden_dim//4, args.hidden_dim*4, args.hidden_dim, norm=BN1D) for _ in range(6)])

        self.num_of_gnn_layer = args.num_of_gnn_layer
        self.agent_gnns = torch.nn.ModuleList([AgentHetGNN(args=args) for _ in range(self.num_of_gnn_layer)])

    def forward(self, input_dic):
        agent_n_emb = self.agent_emb(input_dic, self.shared_coor_encoder)
        agent_n_type_indices = [torch.where((input_dic["graph_lis"].nodes["agent"].data["a_n_type"]) == _) for _ in range(6)]
        agent_n_fea = torch.zeros((agent_n_emb.shape[0], agent_n_emb.shape[1]), device="cuda:"+str(input_dic["gpu"]))
        for _ in range(6):
            agent_n_fea[agent_n_type_indices[_]] = self.temporal_encoders[_](agent_n_emb[agent_n_type_indices[_]])

        agent_edge_types = [('agent', 'self', 'agent'), ('agent', 'other', 'agent')]
        agent_edge_names = ["self", "other"]

        agent_e_type_lis = torch.cat([input_dic["graph_lis"].edata["a_e_type"][_] for _ in agent_edge_types], dim=0)
        agent_e_fea_rel_pos = torch.cat([input_dic["graph_lis"].edata["a_e_fea"][_] for _ in agent_edge_types], dim=0)
        agent_e_type_indices = [torch.where(agent_e_type_lis == _) for _ in range(6)]
        agent_e_src_n_fea = torch.cat([agent_n_fea[input_dic["graph_lis"].edges(etype=_)[0],...] for _ in agent_edge_names], dim=0)
        agent_e_fea_rel_pos = self.shared_rel_encoder(agent_e_fea_rel_pos)
        agent_e_fea = torch.zeros((agent_e_fea_rel_pos.shape[0], self.hidden_dim), device="cuda:"+str(input_dic["gpu"]))
        for _ in range(6):
            agent_e_fea[agent_e_type_indices[_]] = self.agent_e_fea_MLPs[_](torch.cat([agent_e_fea_rel_pos[agent_e_type_indices[_]], agent_e_src_n_fea[agent_e_type_indices[_]]], dim=-1))
        input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"] = agent_n_fea
        agent_e_num_lis_by_etype = np.cumsum([0] + [len(input_dic["graph_lis"].edges[_].data["a_e_type"]) for _ in agent_edge_types])
        for _index, _ in enumerate(agent_edge_types):
            input_dic["graph_lis"].edges[_].data["a_e_hidden"] = agent_e_fea[agent_e_num_lis_by_etype[_index]:agent_e_num_lis_by_etype[_index+1]]

        for i in range(self.num_of_gnn_layer):
            output_agent_n_fea, output_in_agent_e_fea = self.agent_gnns[i](input_dic)
            input_dic["graph_lis"].nodes["agent"].data["a_n_hidden"] = output_agent_n_fea
            input_dic["graph_lis"].edges["self"].data["a_e_hidden"] = output_in_agent_e_fea[0]
            input_dic["graph_lis"].edges["other"].data["a_e_hidden"] = output_in_agent_e_fea[1]
        return input_dic["graph_lis"]


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

    def forward_denoise(self, noisy_traj, t, agent_features, agent_types, context, context_mask):
        """Forward pass of the denoising network."""
        x = self.traj_embed(noisy_traj)
        x = x + self.pos_embed
        
        t_emb = self.time_embed(t)
        type_emb = self.agent_type_embed(agent_types)
        cond_emb = t_emb + type_emb + agent_features
        
        for block in self.transformer_blocks:
            x = block(x, cond_emb, context, context_mask)
        
        x = self.final_norm(x)
        predicted_noise = self.output_proj(x)
        
        return predicted_noise

    def training_forward(self, agent_features, agent_ids, agent_types, labels, input_dic):
        """Training forward pass: add noise and predict it."""
        device = agent_features.device
        batch_size = agent_features.shape[0]
        
        context, context_mask = self.get_context_features(agent_ids, input_dic)
        
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(labels)
        noisy_traj = self.schedule.q_sample(labels, t, noise)
        
        predicted_noise = self.forward_denoise(
            noisy_traj, t, agent_features, agent_types, context, context_mask
        )
        
        return predicted_noise, noise

    @torch.no_grad()
    def ddim_sample(self, agent_features, agent_ids, agent_types, input_dic, num_samples=None, ddim_steps=20):
        """Fast sampling using DDIM (inference only - with torch.no_grad)."""
        if num_samples is None:
            num_samples = self.num_samples
        
        device = agent_features.device
        batch_size = agent_features.shape[0]
        
        context, context_mask = self.get_context_features(agent_ids, input_dic)
        
        agent_features_exp = agent_features.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.hidden_dim)
        agent_types_exp = agent_types.unsqueeze(1).expand(-1, num_samples).reshape(-1)
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
            
            predicted_noise = self.forward_denoise(
                x, t, agent_features_exp, agent_types_exp, context_exp, context_mask_exp
            )
            
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
                
                pred_noise_coef = torch.sqrt(1 - alpha_t_prev)
                x = torch.sqrt(alpha_t_prev) * x_0_pred + pred_noise_coef * predicted_noise
            else:
                x = x_0_pred
        
        trajectories = x.reshape(batch_size, num_samples, self.pred_len, 2)
        return trajectories
    
    def predict_x0_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise (used only during training)."""
        sqrt_alpha_cumprod_t = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t]
        
        while sqrt_alpha_cumprod_t.dim() < x_t.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)
        
        return (x_t - sqrt_one_minus_alpha_cumprod_t * noise) / sqrt_alpha_cumprod_t


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
        
        # Context projection
        self.context_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim // 4, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 5 refinement layers
        self.refine_layers = nn.ModuleList([
            RefineLayer(2, self.hidden_dim // 4, args) 
            for _ in range(self.refine_num)
        ])
    
    def forward(self, pred_traj, obs_traj, agent_features, return_all_steps=False):
        """
        Refine diffusion-generated trajectories.
        
        Args:
            pred_traj: (batch, num_samples, pred_len, 2) - velocities from diffusion
            obs_traj: (batch, obs_len, 2) - observation positions
            agent_features: (batch, hidden_dim) - encoded agent features
            return_all_steps: If True, return all intermediate refinements
        
        Returns:
            If return_all_steps=True: (batch, refine_num+1, num_samples, pred_len, 2)
            If return_all_steps=False: (batch, num_samples, pred_len, 2)
        """
        batch_size, num_samples, pred_len, _ = pred_traj.shape
        
        # Convert velocities to positions
        pred_positions = torch.cumsum(pred_traj, dim=2)
        obs_positions = obs_traj
        obs_exp = obs_positions.unsqueeze(1).expand(-1, num_samples, -1, -1)
        
        # Prepare context (agent features + modal embeddings)
        modal_emb_exp = self.modal_emb.unsqueeze(0).expand(batch_size, -1, -1)
        agent_features_exp = agent_features.unsqueeze(1).expand(-1, num_samples, -1)
        context = torch.cat([agent_features_exp, modal_emb_exp], dim=-1)
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
        
        if return_all_steps:
            # Stack: [initial, refine1, refine2, refine3, refine4, refine5]
            return torch.stack(all_steps, dim=1)  # (batch, 6, num_samples, pred_len, 2)
        else:
            # Return only final refined output (for inference)
            return refined_pred  # (batch, num_samples, pred_len, 2)


# =============================================================================
# Main Model: HDGT with Diffusion + Refiner
# =============================================================================

class HDGT_model(nn.Module):
    """
    HDGT model with diffusion decoder and refiner.
    
    Architecture:
        1. Encoder: HDGT-style heterogeneous graph network
        2. Diffusion Decoder: Generates diverse K samples
        3. Refiner: Polishes samples for accuracy (5 iterations)
    
    Training Losses (CLEAN 2-LOSS):
        1. Noise Loss: Diffusion learns denoising (for diversity)
        2. Multi-Step Refiner Loss: Averaged over all 6 steps (for accuracy)
    """
    def __init__(self, input_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.args = args
        self.num_samples = args.num_prediction
        self.pred_len = args.pred_len
        self.obs_len = args.obs_len
        
        # CLEAN 2-LOSS WEIGHTS (no auxiliary loss!)
        self.noise_loss_weight = 1.0      # Diffusion learns denoising
        self.refiner_loss_weight = 1.0    # Refiner learns positions
        
        # Three components
        self.encoder = HDGT_encoder(input_dim, args)
        self.diffusion_decoder = TransformerDiffusionDecoder(args)
        self.refiner = DiffusionRefiner(args)
        
        self.use_refiner = getattr(args, 'use_refiner', True)
        
    def forward(self, input_dic):
        """
        Forward pass for training and inference.
        
        Training returns: (total_loss, noise_loss, refiner_loss, type_masks, types)
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
            # TRAINING MODE: 2 LOSSES (Noise + Multi-Step Refiner)
            # ═════════════════════════════════════════════════════════════
            labels_vel = input_dic["label_lis"]  # Velocities for diffusion
            labels_pos = input_dic["label_pos_lis"]  # Positions for refiner
            
            # ─────────────────────────────────────────────────────────────
            # LOSS 1: Noise Prediction (Diffusion learns denoising)
            # ─────────────────────────────────────────────────────────────
            predicted_noise, actual_noise = self.diffusion_decoder.training_forward(
                target_agent_features,
                target_agent_ids,
                target_agent_types,
                labels_vel,
                input_dic
            )
            noise_loss = F.mse_loss(predicted_noise, actual_noise)
            
            # ─────────────────────────────────────────────────────────────
            # LOSS 2: Multi-Step Refiner (Like Original HDGT)
            # ─────────────────────────────────────────────────────────────
            refiner_loss = torch.tensor(0.0, device=predicted_noise.device)
            
            if self.use_refiner:
                # Generate K diverse samples (with torch.no_grad for stability)
                with torch.no_grad():
                    diffusion_samples = self.diffusion_decoder.ddim_sample(
                        target_agent_features,
                        target_agent_ids,
                        target_agent_types,
                        input_dic,
                        num_samples=self.num_samples,
                        ddim_steps=20  # High quality samples during training
                    )
                
                obs_trajectories = all_agent_raw_traj[target_agent_ids]
                
                # Get ALL refinement steps (not just final!)
                # Output shape: (batch, 6_steps, K_samples, pred_len, 2)
                all_refined_steps = self.refiner(
                    diffusion_samples,
                    obs_trajectories,
                    target_agent_features,
                    return_all_steps=True  # KEY: Get all intermediate steps
                )
                
                # Compute distance at EVERY step (like original HDGT)
                label_positions_exp = labels_pos.unsqueeze(1).unsqueeze(1).expand(
                    -1, all_refined_steps.shape[1], self.num_samples, -1, -1
                )
                distances = torch.sqrt(
                    ((all_refined_steps - label_positions_exp) ** 2).sum(dim=-1).mean(dim=-1)
                )  # (batch, 6_steps, K_samples)
                
                # Find best mode based on FINAL refinement step
                best_mode_idx = distances[:, -1, :].argmin(dim=-1)  # (batch,)
                
                # Average loss over ALL refinement steps (like original HDGT)
                # This gives equal gradient signal to all refiner layers!
                best_mode_distances = distances[
                    torch.arange(distances.shape[0]),
                    :,  # ALL steps: [0, 1, 2, 3, 4, 5]
                    best_mode_idx
                ]  # (batch, 6_steps)
                
                refiner_loss = best_mode_distances.mean()
            
            # ─────────────────────────────────────────────────────────────
            # TOTAL LOSS: Simple 2-Loss Combination
            # ─────────────────────────────────────────────────────────────
            total_loss = (
                self.noise_loss_weight * noise_loss +      # 1.0
                self.refiner_loss_weight * refiner_loss    # 1.0
            )
            
            return total_loss, noise_loss, refiner_loss, pred_indice_bool_type_lis, target_agent_types
        
        else:
            # ═════════════════════════════════════════════════════════════
            # INFERENCE MODE: Generate diverse predictions
            # ═════════════════════════════════════════════════════════════
            trajectories = self.diffusion_decoder.ddim_sample(
                target_agent_features,
                target_agent_ids,
                target_agent_types,
                input_dic,
                num_samples=self.num_samples,
                ddim_steps=20  # High quality for inference
            )
            
            if self.use_refiner:
                obs_trajectories = all_agent_raw_traj[target_agent_ids]
                trajectories = self.refiner(
                    trajectories, 
                    obs_trajectories, 
                    target_agent_features,
                    return_all_steps=False  # Only return final
                )
            
            # Flag to indicate output is positions (for metric calculation)
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
