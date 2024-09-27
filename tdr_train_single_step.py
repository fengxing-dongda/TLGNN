import argparse
import torch.optim as optim
from tdr_net import *
from tdr_util import *
import torch.nn as nn
parser = argparse.ArgumentParser(description='PyTorch Multivariate Time series forecasting')
parser.add_argument('--data', type=str, default='./data/electricity.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=321,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--node_k',type=int,default=60,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--input_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--dilated_channels',type=int,default=16,help='dilated channels')
parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--Lagmax', type=int, default=3)
parser.add_argument('--sliding_interval', type=int, default=2)
parser.add_argument('--layers',type=int,default=5,help='number of layers')
parser.add_argument('--batch_size',type=int,default=4,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')
args = parser.parse_args()
device = torch.device(args.device)
def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(3,4)
        if iter % args.step_size == 0:
            perm = range(args.num_nodes * (args.Lagmax+1))
            id = range(args.num_nodes)
        for j in range(args.num_split):
            id = torch.tensor(id).to(device)
            perm = torch.tensor(perm).to(device)
            tx = X[:, :, :, :, :]
            ty = Y[:, :]
            output = model(tx)
            print(output)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:,:]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()
    return total_loss / n_samples
class Optim(object):
    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._makeOptimizer()

    def step(self):
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()
        return  grad_norm

    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True
        if self.start_decay:
            self.lr = self.lr * self.lr_decay
        self.start_decay = False
        self.last_ppl = ppl
        self._makeOptimizer()
def main():
    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.Lagmax, args.normalize, args.sliding_interval)
    model = TLGNN(args.gcn_depth, args.num_nodes,args.Lagmax + 1, args.node_k, args.node_dim, args.dilation_exponential,
                  args.device, args.input_dim, args.dilated_channels, dropout=args.dropout, propalpha=args.propalpha, layers=args.layers,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels, skip_channels=args.skip_channels,
                  end_channels= args.end_channels, seq_length = args.seq_in_len, in_dim=args.in_dim,
                   out_dim=args.seq_out_len, layer_norm_affline=False)
    model = model.to(device)
    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            print(train_loss)
    except KeyboardInterrupt:
        print('Exiting from training early')

if __name__ == "__main__":
    main()
