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
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.gc = tdr_graph_constructor(num_nodes, Lagmax, node_k, node_dim, device)
        self.seq_length = seq_length
        self.layers = layers
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

        self.start_conv = nn.Conv2d(in_channels= input_channels,
                                    out_channels= residual_channels,
                                    kernel_size=(1, 1))
        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)
                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor = new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor = new_dilation))
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
                self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                new_dilation *= dilation_exponential
    def forward(self, input, idx=None):
        batch_size = input.size(0)
        Lagmax = input.size(2)
        n = input.size(3)
        seq_len= input.size(4)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))
            seq_len = self.receptive_field
        if idx is None:
            adp = self.gc(self.idx)
        else:
            adp = self.gc(idx)
        x = torch.zeros((batch_size, 1, n * Lagmax, seq_len))
        for j in range(n):
            for k in range(Lagmax):
                x[:, :, j*Lagmax + k,:] = input[:, :, k, j, :]
        x = x.to(self.device)
        skip = self.skip0(F.dropout(x, self.dropout, training=self.training))
        x = self.start_conv(x)
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            x = self.gconv1[i](x, adp)
            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)
        skip = self.skipE(x) + skip
        out = torch.zeros((batch_size, 32, n , 1))
        for j in range(n):
            out[:, :, j,:] = skip[:, :, (j+1) * self.Lagmax-1, :]
        out = out.to(self.device)
        out = F.relu(out)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)
        return out






