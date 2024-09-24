from tdr_layer import *
import torch.nn as nn
class TLGNN(nn.Module):
    def __init__(self, gcn_depth, num_nodes, Lagmax, node_k, node_dim, dilation_exponential, device, input_channels, dilated_channels, dropout = 0.3, propalpha=0.05,layers = 2,  residual_channels = 16, conv_channels = 32, skip_channels = 32,end_channels=128,in_dim=2, out_dim=12, seq_length = 168, layer_norm_affline=True):
        super(TLGNN, self).__init__()
        self.gcn_depth = gcn_depth
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.device = device
        self.layers = layers
        self.Lagmax = Lagmax
        self.idx = torch.arange(self.num_nodes*self.Lagmax).to(device)
        # 空洞卷积
        self.filter_convs = nn.ModuleList()
        # 门控部分
        self.gate_convs = nn.ModuleList()
        # 残差网络
        self.residual_convs = nn.ModuleList()
        # 跳跃连接
        self.skip_convs = nn.ModuleList()
        # 图卷积1
        self.gconv1 = nn.ModuleList()
        self.norm = nn.ModuleList()
        # 图构造 graph_constructor 继承自 nn.module subgraph_size 保存前 k 个节点
        self.gc = tdr_graph_constructor(num_nodes, Lagmax, node_k, node_dim, device)
        # 序列长度
        self.seq_length = seq_length
        # 空洞卷积的层数
        self.layers = layers
        # 空洞卷积里面卷积核的长度
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        self.end_linear = nn.Linear(16, 1)

        # 开始的卷积层 输入数据输入之后先做一个2d卷积
        self.start_conv = nn.Conv2d(in_channels= input_channels,
                                    out_channels= residual_channels,
                                    kernel_size=(1, 1))
        for i in range(1):
            if dilation_exponential > 1:
                # 计算感受野的大小
                rf_size_i = int(1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            # 膨胀因子
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    # 在哪一层的感受野  j 表示第几层
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)
                # 空洞卷积层
                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor = new_dilation))
                # 门控层
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor = new_dilation))
                # # 残差网络层 在不使用图卷积的时候使用它
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))
                # 修改了
                self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                new_dilation *= dilation_exponential
        # 图卷积部分不是每一层都有的
    # 输入 本次训练的部分子节点
    def forward(self, input, idx=None):

        # 真实输入序列的长度  输入为 input (batch_size, 1, Lagmax+1, 137, 168)
        batch_size = input.size(0)
        Lagmax = input.size(2)
        n = input.size(3)
        seq_len= input.size(4)
        # 输入长度和预先定义的长度不一样
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        # 如果输入长度小于预设长度(感受野) 补零操作 self.receptive_field = 187
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))
            seq_len = self.receptive_field
        # # 根据索引生成 图的邻接矩阵
        if idx is None:
            adp = self.gc(self.idx)
        else:
            adp = self.gc(idx)
        # x = torch.squeeze(input,2)
        # 整理数据
        x = torch.zeros((batch_size, 1, n * Lagmax, seq_len))
        for j in range(n):
            for k in range(Lagmax):
                x[:, :, j*Lagmax + k,:] = input[:, :, k, j, :]
        x = x.to(self.device)
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        # 将输入先经过一个2d 的cnn x
        x = self.start_conv(x)
        # 进行多次空洞卷积
        for i in range(self.layers):
            # 残差
            residual = x
            # 第一部分  进行特征提取 filter_convs是一个数组 每次训练对应的一部分
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            # 在skip输出之前用图卷积
            # s = self.gconv1[i](s, adp)
            s = self.skip_convs[i](s)
            skip = s + skip
            # 在整个网络上进行图卷积
            x = self.gconv1[i](x, adp)
            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)
        skip = self.skipE(x) + skip
        # 图卷积
        # x = self.gconv1[0](x, adp)
        # x2 = x.cpu().detach().numpy()
        # out = skip
        out = torch.zeros((batch_size, 32, n , 1))
        for j in range(n):
            out[:, :, j,:] = skip[:, :, (j+1) * self.Lagmax-1, :]
        out = out.to(self.device)
        # 原始的MTGNN中的预测模型  激活函数+卷积+激活+卷积
        out = F.relu(out)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)
        # out2 = out.cpu().detach().numpy()
        return out






