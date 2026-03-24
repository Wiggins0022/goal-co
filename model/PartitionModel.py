import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, edge_feats=1, units=48, act_fn='silu', agg_fn='mean'):  # TODO feats=1
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(edge_feats, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            #x0是原始节点的特征
            x0 = x
            #节点自信息
            x1 = self.v_lins1[i](x0)
            #邻居节点信息
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            #边的自信息
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            # 使用agg_fn采用平均汇聚，将x2[edge_index[1]]的信息汇聚到每个节点上，与原来节点的信息相加，再经过批量归一化、silu
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            #更新边特征信息
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w



class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        #主线性层列表
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        #专门处理首节点信息的线性层列表
        self.lin_f = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        #专门处理尾结点信息的线性层列表
        self.lin_l = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        #专门处理已访问节点信息的线性层列表
        self.lin_v = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def forward(self, node, x, solution, visited, k_sparse):
        sample_size = solution.size(0)
        x_first = node.clone()[None, :].repeat(solution.size(0), 1, 1).gather(1, solution[:, 0][:, None, None].expand(-1, -1, node.size(-1)))
        x_last = node.clone()[None, :].repeat(solution.size(0), 1, 1).gather(1, solution[:, -1][:, None, None].expand(-1, -1, node.size(-1)))
        x_visited = node.clone()
        for i in range(self.depth):
            x = self.lins[i](x)
            x_first = self.lin_f[i](x_first)
            x_last = self.lin_l[i](x_last)
            x_visited = self.lin_v[i](x_visited)
            if i < self.depth - 1:
                x = self.act_fn(x)
                x_first = self.act_fn(x_first)
                x_last = self.act_fn(x_last)
                x_visited = self.act_fn(x_visited)
            else:
                #q只保留被标记为已访问节点的信息，其他节点特征被置为0，q表示当前已选择的节点的整体特征
                q = x_visited[None, :].repeat(solution.size(0), 1, 1) * visited[:, :, None].expand(-1, -1, node.size(-1))
                #对节点的特征平均值
                q = q.sum(1) / visited[:, :, None].expand(-1, -1, node.size(-1)).sum(1)
                q += x_first.squeeze(1) + x_last.squeeze(1)
                #计算出当前路径上的节点与所有节点的相关性，保留最重要的k_sparse个节点的相关性
                x = torch.mm(q, x.T).reshape(sample_size, -1, k_sparse)
                x = torch.softmax(x, dim=-1)
        return x



class ParNet(MLP):
    def __init__(self, k_sparse, depth=3, units=48, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        self.k_sparse = k_sparse
        super().__init__([self.units] * depth, act_fn)

    def forward(self, x_emb, e_emb, solution, visited):
        return super().forward(x_emb, e_emb, solution, visited, self.k_sparse).squeeze(dim=-1)


class PartitionModel(nn.Module):
    def __init__(self, units, feats, k_sparse, edge_feats=1, depth=12):
        super().__init__()
        self.emb_net = EmbNet(depth=depth, units=units, feats=feats, edge_feats=edge_feats)
        self.par_net_heu = ParNet(units=units, k_sparse=k_sparse)
        self.x_emb = None

    def pre(self, pyg):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        self.x_emb, self.emb = self.emb_net(x, edge_index, edge_attr)

    def forward(self, solution=None, visited=None):
        solution_cat = torch.cat((solution[:, 0].unsqueeze(-1), solution[:, -1].unsqueeze(-1)), dim=-1)
        heu = self.par_net_heu(self.x_emb, self.emb, solution_cat, visited)
        return self.x_emb, heu

    @staticmethod
    def reshape(pyg, vector):
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(vector.size(0), n_nodes, n_nodes), device=device)
        idx = torch.repeat_interleave(torch.arange(vector.size(0)).to(device), repeats=pyg.edge_index[0].shape[0])
        idx0 = pyg.edge_index[0].repeat(vector.size(0))
        idx1 = pyg.edge_index[1].repeat(vector.size(0))
        matrix[idx, idx0, idx1] = vector.view(-1)
        try:
            assert (matrix.sum(dim=2) >= 0.99).all()
        except:
            torch.save(matrix, './error_reshape.pt')
        return matrix
