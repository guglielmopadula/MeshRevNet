import torch
from torch import nn
import numpy as np


class ActNorm(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.bias=nn.parameter.Parameter(torch.zeros(self.input_size))
        self.a=nn.parameter.Parameter(torch.ones(self.input_size))

    def forward(self, x):
        ndim=x.ndim 
        x=(self.a.view(*np.ones(ndim-1,dtype=np.int64),-1))*x+self.bias.view(*np.ones(ndim-1,dtype=np.int64),-1)
        return x
    
    def inverse(self, y):
        ndim=y.ndim 
        y=(y-self.bias.view(*np.ones(ndim-1,dtype=np.int64),-1))/self.a.view(*np.ones(ndim-1,dtype=np.int64),-1)
        return y
    
    

class OneSizeConv(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lin=nn.Linear(self.input_size,self.output_size,bias=False)


    def forward(self, x):
        return self.lin(x)
    
    def inverse(self, x):
        pinv=torch.pinverse(self.lin.weight.t())
        return x@pinv
    
class AffineCoupling(nn.Module):
    def __init__(self,output_size,intermediate_size):
        super().__init__()
        self.output_size = output_size
        self.intermediate_size = intermediate_size
        self.nn=nn.Sequential(nn.Linear(self.intermediate_size,self.intermediate_size),nn.BatchNorm1d(intermediate_size),nn.Sigmoid(),
                              nn.Linear(self.intermediate_size,self.intermediate_size),nn.BatchNorm1d(intermediate_size),nn.Sigmoid(),
                              nn.Linear(self.intermediate_size,self.intermediate_size),nn.BatchNorm1d(intermediate_size),nn.Sigmoid(),
                              nn.Linear(self.intermediate_size,self.intermediate_size),nn.BatchNorm1d(intermediate_size),nn.Sigmoid(),
                              nn.Linear(self.intermediate_size,self.intermediate_size),nn.BatchNorm1d(intermediate_size),nn.Sigmoid(),
                              nn.Linear(self.intermediate_size,2*(self.output_size-self.intermediate_size)))
        self.perm=torch.randperm(self.output_size)
    
    def forward(self, x):
        x=x[...,self.perm]
        x_a,x_b=torch.split(x,[self.output_size-self.intermediate_size,self.intermediate_size],dim=-1)
        s,t=torch.split(self.nn(x_b),[self.output_size-self.intermediate_size,self.output_size-self.intermediate_size],dim=-1)
        s=torch.exp(s)
        y_a=s*x_a+t
        y_b=x_b
        return torch.cat([y_a,y_b],dim=-1)
    
    def inverse(self, y):
        y_a,y_b=torch.split(y,[self.output_size-self.intermediate_size,self.intermediate_size],dim=-1)
        s,t=torch.split(self.nn(y_b),[self.output_size-self.intermediate_size,self.output_size-self.intermediate_size],dim=-1)
        s=torch.exp(s)
        x_a=(y_a-t)/s
        x_b=y_b
        x=torch.cat([x_a,x_b],dim=-1)
        x=x[...,torch.argsort(self.perm)]
        return x

class RevNetLayer(nn.Module):
    def __init__(self, input_size,output_size,intermediate_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.intermediate_size = intermediate_size
        self.actnorm=ActNorm(self.input_size)
        self.one_size_conv=OneSizeConv(self.input_size,self.output_size)
        self.affine_coupling=AffineCoupling(self.output_size,self.intermediate_size)


    def forward(self, x):
        #Step 1
        x=self.actnorm(x)
        #Step 2
        x=self.one_size_conv(x)
        #Step 3
        x=self.affine_coupling(x)
        return x

    def inverse(self, y):
        #Step 3
        x=self.affine_coupling.inverse(y)
        #Step 2
        x=self.one_size_conv.inverse(x)
        #Step 1
        x=self.actnorm.inverse(x)
        return x
    
class RevNet(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.layers=nn.ModuleList([
            RevNetLayer(input_size,500,100),
            RevNetLayer(500,output_size,100)
            ])
    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return x

    def inverse(self, y):
        for layer in reversed(self.layers):
            y=layer.inverse(y)
        return y


