from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F

class dilated_inception(nn.Module):

    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))
    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x
        
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()
        
class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        # adj2 = adj.cpu().detach().numpy()
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() 

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:  
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.einsum('ncwl,lm->ncwm', (input_feature, self.weight))
        output = torch.einsum('ncwl,vw->ncvl', (support , adjacency))
        
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'

class GcnNet(nn.Module):

    def __init__(self, input_dim, output_dim, use_bias):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, output_dim, use_bias)
        self.gcn2 = GraphConvolution(input_dim, output_dim, use_bias)

    def forward(self,x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        adj2 = adj.cpu().detach().numpy()
        d = adj.sum(1)
        d2 = d.cpu().detach().numpy()
        a = adj / d.view(-1, 1)
        a2 = a.cpu().detach().numpy()
        h = F.relu(self.gcn1(a, x))  # (N,1433)->(N,16)
        h = F.relu(self.gcn2(a, h))
        return h


class tdr_graph_constructor(nn.Module):
    def __init__(self, nnodes, Lagmax, k, dim, device, alpha=3):
        super(tdr_graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.Lagmax = Lagmax
        self.emb1 = nn.Embedding(nnodes * Lagmax, dim)
        self.emb2 = nn.Embedding(nnodes * Lagmax, dim)
        self.lin1 = nn.Linear(dim,dim)
        self.lin2 = nn.Linear(dim,dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
    def forward(self, idx):
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        adj = adj + torch.rand_like(adj) * 0.01
        mask_peak = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask_peak.fill_(float('0'))
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





