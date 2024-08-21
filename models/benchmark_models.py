import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from lib.lorentz.layers import LorentzFullyConnected, LorentzConv1d, LorentzBatchNorm1d, LorentzReLU, LorentzGlobalAvgPool2d, LorentzMLR
from lib.lorentz.manifold import CustomLorentz

def get_euclidean_convolution_block(channels_sizes,
                                    kernel_size: int = 9,
                                    padding: int = 4):
    return nn.Sequential(
            nn.Conv1d(channels_sizes[0], channels_sizes[1], kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels_sizes[2], channels_sizes[3], kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels_sizes[3]))


def get_hyperbolic_convolution_block(manifold, 
                                     channels_sizes,
                                     kernel_size: int = 9,
                                     padding: int = 4):
    return nn.Sequential(
            LorentzConv1d(manifold=manifold, in_channels=channels_sizes[0], out_channels=channels_sizes[1], kernel_size=kernel_size, padding=padding),
            LorentzBatchNorm1d(manifold=manifold, num_features=channels_sizes[1]),
            LorentzReLU(manifold=manifold),
            LorentzConv1d(manifold=manifold, in_channels=channels_sizes[2], out_channels=channels_sizes[3], kernel_size=kernel_size, padding=padding),
            LorentzBatchNorm1d(manifold=manifold, num_features=channels_sizes[3])
            )


class HyperbolicCNN(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 length: int, 
                 model_dim: int, 
                 fc_dim: int,
                 num_layers: int = 3, 
                 multi_k_model: bool = False,
                 learnable_k: bool = True,
                 k: float = 1.0):
        super(HyperbolicCNN, self).__init__()

        self.multi_k_model = multi_k_model
        self.output_length = length
        self.num_layers = num_layers
        self.manifolds = nn.ModuleList([CustomLorentz(k=k, learnable=learnable_k)])
        if self.multi_k_model:
            self.manifolds.extend([CustomLorentz(k=k, learnable=learnable_k) for _ in range(self.num_layers)])
        else:
            self.manifolds.extend([self.manifolds[0] for _ in range(self.num_layers)])
        
        initial_channel_sizes = [6] + [model_dim] * 3
        channel_sizes = [model_dim] * 4
        
        self.conv_layers = nn.ModuleList([
            get_hyperbolic_convolution_block(self.manifolds[0], initial_channel_sizes),
        ])
        self.conv_layers.extend([get_hyperbolic_convolution_block(self.manifolds[i + 1], channel_sizes) for i in range(self.num_layers - 1)])
        
        self.shortcut = nn.Sequential()
        
        self.activations = nn.ModuleList([
            LorentzReLU(manifold=manifold) for manifold in self.manifolds
        ])
        
        self.fc_layer = LorentzFullyConnected(
            manifold=self.manifolds[-1],
            in_features=1 + ((model_dim - 1) * self.output_length),
            out_features=fc_dim,
            bias=True
        )
        
        self.mlr = LorentzMLR(
            manifold=self.manifolds[-1],
            num_features=fc_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.pad(x, pad=(1, 0))
        x = self.manifolds[0].projx(x)
        
        for i in range(self.num_layers):
            out = self.conv_layers[i](x)
            if i > 0:
                out = out.narrow(-1, 1, res.shape[-1]-1) + res.narrow(-1, 1, res.shape[-1]-1)
                out = self.manifolds[i].add_time(out)
            x = self.activations[i](out)
            if self.multi_k_model:
                x = self.manifolds[i+1].expmap0(self.manifolds[i].logmap0(x))
            res = self.shortcut(x)
        
        x = x.unsqueeze(2) #[b, l, c] -> [b, l, 1, c]
        x = self.manifolds[-1].lorentz_flatten(x)
        x = self.fc_layer(x)
        x = self.activations[-1](x)
        x = self.mlr(x)
        
        return x



class EuclideanCNN(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 length: int, 
                 model_dim: int, 
                 fc_dim: int,
                 num_layers: int = 3,
                 ):
        super(EuclideanCNN, self).__init__()
        
        self.output_length = length
        self.num_layers = num_layers
        initial_channel_sizes = [5] + [model_dim] * 3
        channel_sizes = [model_dim] * 4
        
        self.conv_layers = nn.ModuleList([
            get_euclidean_convolution_block(initial_channel_sizes)
        ])
        self.conv_layers.extend([get_euclidean_convolution_block(channel_sizes) for _ in range(self.num_layers - 1)])
    
        self.activations = nn.ModuleList([
            nn.ReLU(inplace=True) for _ in range(self.num_layers + 1)
        ])        
        
        self.fc_layer = nn.Linear(self.output_length * model_dim, fc_dim)
        self.mlr = nn.Linear(fc_dim, num_classes)
        
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)         
            if i > 0:
                x += out
            x = self.activations[i](x)
            out = x

        x = x.view(x.shape[0], -1)       
        x = self.fc_layer(x)
        x = self.activations[-1](x)
        x = self.mlr(x)
        return x


