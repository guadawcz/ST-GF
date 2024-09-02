import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.tools import normalize
from abc import abstractmethod
from math import sqrt
from utils.init import glorot_weight_zero_bias
EOS = 1e-10

class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class SpatialGraph(nn.Module):
    def __init__(self,
                 n_nodes=22,
                 adj=None,
                 k=2,
                 spatial_GCN=True,
                 device=0):
        super(SpatialGraph, self).__init__()
        self.xs, self.ys = torch.tril_indices(n_nodes, n_nodes, offset=-1)
        node_value = adj[self.xs, self.ys]
        self.edge_weight = nn.Parameter(node_value.clone().detach(), requires_grad=spatial_GCN)
        self.n_nodes = n_nodes
        self.spatial_GCN = spatial_GCN
        self.k = k
        self.device = device

    def forward(self, x):
        if not(self.spatial_GCN):
            edge_weight = torch.eye(self.n_nodes,self.n_nodes)
        else:
            edge_weight = torch.zeros([self.n_nodes, self.n_nodes], device=self.device)
            edge_weight[self.xs.to(self.device), self.ys.to(self.device)] = self.edge_weight.to(self.device)
            edge_weight = edge_weight + edge_weight.T + torch.eye(self.n_nodes, dtype=edge_weight.dtype, device=self.device)
            edge_weight = normalize(edge_weight)
            edge_weight = edge_weight.cuda(self.device)
            edge_weight = edge_weight+torch.eye(self.n_nodes,self.n_nodes).cuda(self.device) 

        x = x.permute(0,2,1,3)
        x = x.contiguous().view(x.shape[0],self.n_nodes,-1)
        for k in range(self.k):
            edge_weight = edge_weight.to(self.device)
            x = torch.matmul(edge_weight,x)
        return x,edge_weight


class TimeGraph(nn.Module):
    def __init__(self, window, k, channels,time_GCN=True):
        super(TimeGraph,self).__init__()
        self.adj = nn.Parameter(0.5*torch.ones(window,window)+1.5*torch.eye(window,window), requires_grad=time_GCN)
        self.time_GCN = time_GCN
        self.window = window
        self.channels = channels
        self.k = k
    
    def forward(self,x):
        if not(self.time_GCN):
            A = torch.eye(self.window,self.window)
            adj = A
        else:
            adj = self.adj
            adj = (adj + adj.T)/2
            A = normalize(adj)
            adj = F.relu(adj)

        x = x.permute(0,2,1,3)
        x = x.contiguous().view(x.shape[0],x.shape[1],-1)

        for i in range(self.k):
            A = A.cuda(x.device)
            x = torch.matmul(A,x)

        x = x.view(x.shape[0],x.shape[1],self.channels,-1)
        x = x.permute(0,2,1,3)

        return x,adj

class GraphAttention(nn.Module):
    def __init__(self,nodes,time_length):
        super(GraphAttention,self).__init__()
        self.nodes = nodes
        self.adj = torch.ones(nodes,nodes)
        self.layernorm = nn.LayerNorm([nodes,time_length],eps=1e-2)
        self.leakyrelu = nn.LeakyReLU(0.0001)
        self.a = nn.Parameter(torch.empty(size=(2*time_length,1)),requires_grad=True)
        nn.init.xavier_normal_(self.a.data,gain=1.414)
        self.node_weight = nn.Parameter(torch.ones(nodes),requires_grad=True)

    def forward(self,x):

        adj = self.adj
        I = torch.eye(22).unsqueeze(0).expand(x.shape[0],-1,-1)

        x_out = []
        x = torch.split(x,self.nodes,dim=2)

        L1_Loss = torch.sum(self.node_weight)

        for t in x:
            temp = t
            temp = temp.permute(0,2,1,3)
            temp = temp.contiguous().view(x.shape[0],self.nodes,-1)

            e = self.prepare_batch(temp)

            zero_vec = (0* torch.ones_like(e)).cuda(x.device)
            e = e.cuda(x.device)
            adj = adj.cuda(x.device)

            attention = torch.where(adj > 0, e, zero_vec)     

            attention = F.softmax(attention,dim=-1)

            attention = F.dropout(attention,0.2)
            
            
            attention = torch.matmul(torch.diag(self.node_weight).repeat(temp.shape[0],1,1),torch.matmul(attention,torch.diag(self.node_weight))) + \
            torch.matmul(torch.diag(self.node_weight).repeat(temp.shape[0],1,1),torch.matmul(10*I.cuda(1),torch.diag(self.node_weight)))

            temp = torch.matmul(attention,temp)
            x_out.append(temp)

        x_out = torch.stack(x_out)
        x = torch.cat((x_out[0],x_out[1]),dim=1)

        return x,attention[0],L1_Loss

class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0,
                                         maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class STGENET(BaseModel):
    def __init__(self,
                 Adj,
                 in_chans,
                 n_classes,
                 time_window_num,
                 spatial_GCN,
                 time_GCN,
                 k_spatial,
                 k_time,
                 dropout,
                 input_time_length=125,
                 out_chans=64,
                 kernel_size=63,
                 slide_window=8,
                 sampling_rate=250,
                 device=0,
                 ):
        super(STGENET, self).__init__()

        self.__dict__.update(locals())
        del self.self

        self.device = device

        self.time_window_num = time_window_num

        self.spatialconv = Conv(nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False, groups = 1),
                          bn=nn.BatchNorm1d(out_chans), activation=None)
        
        self.timeconv = nn.ModuleList()
        self.layers=1
        for _ in range(self.layers):
            self.timeconv.append(Conv(nn.Conv1d(out_chans, out_chans, kernel_size, 1, padding='same',bias=False),
                                   bn=nn.BatchNorm1d(out_chans), activation=None))

        self.downSampling = nn.AvgPool1d(int(sampling_rate//2),int(sampling_rate//2))
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(out_chans*(input_time_length*slide_window//(sampling_rate//2)),n_classes))

        self.ge = nn.Sequential(
            SpatialGraph(n_nodes=in_chans, adj=Adj, k=k_spatial, spatial_GCN=spatial_GCN, device=device),
        )

        self.ge_space = nn.Sequential(TimeGraph(slide_window,k_time,out_chans,time_GCN))

        # ##########################another module If you want to change another please watch this.###############################################
        # GAT and some conv have been prepared.
        self.GAT = GraphAttention(nodes=in_chans,time_length=input_time_length*slide_window)
        self.attention = nn.MultiheadAttention(64,8,batch_first=True)

        self.conv = nn.Sequential(
            Conv2dWithConstraint(in_channels = 32, out_channels = 32, kernel_size=(1,1),max_norm=None,stride=1,bias = False),
            nn.BatchNorm2d(32,momentum=0.01, affine=True, eps=1e-3)
        )
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(32, 64, (22, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=32, padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True,
                           eps=1e-3),)
        
        self.temporal_conv = nn.Sequential(
            Conv2dWithConstraint(in_channels=1, out_channels=8,
                                 kernel_size=(1, 64),
                                 max_norm=None,
                                 stride=1,
                                 bias=False,
                                 padding='same'
                                 ),)
        self.separable_conv = nn.Sequential(
            Conv2dWithConstraint(64, 64, (1, 16),
                                 max_norm=None,
                                 stride=1,
                                 bias=False, groups=64,
                                 padding=(0, 8)),
            Conv2dWithConstraint((64), 16, (1, 1), max_norm=None, stride=1, bias=False,
                                 padding=(0, 0)),
            nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3),)
        self.cls = nn.Sequential(
            Conv2dWithConstraint(16, self.n_classes,
                                 (1125, 1), max_norm=0.25,
                                 bias=True),)
       ############################################################################################################################
        self.apply(glorot_weight_zero_bias)



    def forward(self, x):
        x = x.view(x.shape[0],x.shape[1],self.time_window_num,-1).permute(0,2,1,3)

        x,node_weights = self.ge(x)

        x = self.spatialconv(x)

        for i in range(len(self.timeconv)):
            x = self.timeconv[i](x)

        x = F.gelu(x)
        x = x.contiguous().view(x.shape[0],x.shape[1],self.slide_window,-1)

        x,space_node_weights = self.ge_space(x)

        x = x.contiguous().view(x.shape[0],x.shape[1],-1)
        x = self.downSampling(x)
        x = self.dp(x)

        x = x.view(x.shape[0],-1)
        x = x.cuda(self.device)

        x = self.fc(x)

        return x,node_weights,space_node_weights
