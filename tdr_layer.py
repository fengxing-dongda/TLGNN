from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F

# 空洞卷积层
class dilated_inception(nn.Module):
    # dilation_factor 膨胀因子
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        # 生成进行卷积的数组
        self.tconv = nn.ModuleList()
        # 卷积核大小的设置
        self.kernel_set = [2,3,6,7]
        # 每一个卷积核对应的cout 输出 总的输出长度除以采用的卷积核的数量
        cout = int(cout/len(self.kernel_set))
        # 遍历卷积核数组中的每一个尺寸的卷积核 生成卷积数组
        for kern in self.kernel_set:
            # 卷积核 添加
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))
    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            # 使用每一个卷积核进行卷积
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            # 这个操作是统一长度  因为 卷积核size不一样 所以 卷积后的长度会稍微有一些不一样
            x[i] = x[i][...,-x[-1].size(3):]
        # concatenate 操作
        x = torch.cat(x, dim=1)
        return x
# 每一层的深度学习模型
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()
# 线性层
class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        # nconv()
        self.nconv = nconv()
        # 线性层  也就是 mlp。 输入的维度为图卷积的层数加一，
        # 输入的数据为最初的数据H0和每次图卷积之后的数据Hk
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        # adj 邻接矩阵 + 度矩阵
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        # adj2 = adj.cpu().detach().numpy()
        d = adj.sum(1)
        # d2 = d.cpu().detach().numpy()
        h = x
        out = [h]
        # 最终的拉普拉斯矩阵
        a = adj / d.view(-1, 1)
        # a2 = a.cpu().detach().numpy()
        # gcn 的层数
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            # 每次进行完图卷积之后 都放到一个数组中
            out.append(h)
        # 连接起来
        ho = torch.cat(out,dim=1)
        # 经过一个线性层
        ho = self.mlp(ho)
        return ho

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        From https://blog.csdn.net/sdu_hao/article/details/104143731
        Args:
        ----------
            input_dim: int
                节点输入特征的维度 D
            output_dim: int
                输出特征维度 D‘
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # 定义GCN层的权重矩阵
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 使用自定义的参数初始化方式

    def reset_parameters(self):
        # 自定义参数初始化方式
        # 权重参数初始化方式
        init.kaiming_uniform_(self.weight)
        if self.use_bias:  # 偏置参数初始化为0
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        # input_feature(batchsize,通道数,)
        support = torch.einsum('ncwl,lm->ncwm', (input_feature, self.weight))
        output = torch.einsum('ncwl,vw->ncvl', (support , adjacency))
        # support = torch.mm(input_feature, self.weight)  # XW (N,D');X (N,D);W (D,D')
        # output = torch.sparse.mm(adjacency, support)  # (N,D')
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'

class GcnNet(nn.Module):
    """
    From https://blog.csdn.net/sdu_hao/article/details/104143731
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, input_dim, output_dim, use_bias):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, output_dim, use_bias)
        self.gcn2 = GraphConvolution(input_dim, output_dim, use_bias)

    def forward(self,x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        adj2 = adj.cpu().detach().numpy()
        d = adj.sum(1)
        d2 = d.cpu().detach().numpy()
        # 输入对应论文中的  H
        # 得到归一化的拉普拉斯矩阵
        a = adj / d.view(-1, 1)
        a2 = a.cpu().detach().numpy()
        h = F.relu(self.gcn1(a, x))  # (N,1433)->(N,16)
        h = F.relu(self.gcn2(a, h))
        return h

# 图构造模块 图学习模块
class tdr_graph_constructor(nn.Module):
    # 节点数量，最大滞后时间，保留前k个模块，嵌入维度，算法运行设备，学习率
    def __init__(self, nnodes, Lagmax, k, dim, device, alpha=3):
        super(tdr_graph_constructor, self).__init__()
        # 节点数量
        self.nnodes = nnodes
        # 最大滞后时间
        self.Lagmax = Lagmax
        # nn.embedding就是一个简单的查找表，存储固定字典和大小的嵌入。生成的单词表中词的数量为nnodes * Lagmax
        self.emb1 = nn.Embedding(nnodes * Lagmax, dim)
        self.emb2 = nn.Embedding(nnodes * Lagmax, dim)
        self.lin1 = nn.Linear(dim,dim)
        self.lin2 = nn.Linear(dim,dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
    # 根据节点的向量嵌入表示 生成图
    def forward(self, idx):
        # 这里要改一下
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)
        # 节点向量表示  对应论文中 M1 M2 对应于
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
        # 这里只把 M1 和 M2的转置进行处理
        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        # 做一个简单处理，避免全是0 或者 0 过多影响 topk函数的计算
        adj = adj + torch.rand_like(adj) * 0.01
        # 确定滞后时间的代码
        mask_peak = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask_peak.fill_(float('0'))
        # torch.zeros()函数，返回一个形状为为size, 类型为torch.dtype，里面的每一个值都是0的tensor
        # 生成一个mask矩阵
        mask_k = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask_k.fill_(float('0'))
        for i in range(0, self.nnodes):
            for j in range(0, self.nnodes):
                c = adj[i * self.Lagmax + (self.Lagmax - 1), j * self.Lagmax:(j + 1) * self.Lagmax]
                top_peak_values, top_peak_indices = c.topk(1)
                mask_peak[i * self.Lagmax + (self.Lagmax - 1), (j * self.Lagmax + top_peak_indices)] = 1
        adj = adj * mask_peak
        for i in range(0, self.nnodes):
            d = adj[i * self.Lagmax + (self.Lagmax - 1), :]
            top_k_values, top_k_indices = d.topk(self.k)
            mask_k[i * self.Lagmax + (self.Lagmax - 1), top_k_indices] = 1
        adj = adj * mask_k
        return adj

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()



    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)





