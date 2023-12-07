import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


from methods.meta_template import MetaTemplate


class FSCT(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(FSCT, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        
        dim = self.feat_dim
        self.ATTN = Attention(dim)
        self.sm = nn.Softmax(dim = -2)
        self.proto_weight = nn.Parameter(torch.ones(n_way, n_support, 1))
        
        self.FFN = nn.Sequential(nn.LayerNorm(dim),
                                 nn.Linear(dim, 512),
                                 nn.GELU(),
                                 nn.Linear(512, dim))
        
        self.linear = nn.Sequential(nn.LayerNorm(dim),
                                    nn.Linear(dim, 64),
                                    CosineDistLinear(64, 1))


    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        z_proto = (z_support.view(self.n_way, self.n_support, -1)* self.sm(self.proto_weight)).sum(1).unsqueeze(0)  # the shape of z is [1, n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)
        
        x = self.ATTN(q = z_proto, k = z_query, v = z_query) + z_proto
        x = self.FFN(x) + x
        
        scores = self.linear(x).squeeze()

        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.heads = 8
        self.dim_head = 64
        inner_dim = self.heads * self.dim_head
        self.scale = self.dim_head ** -0.5
        self.sm = nn.Softmax(dim = -1)
        
        self.input_linear = nn.Sequential(nn.LayerNorm(dim),
                                          nn.Linear(dim, inner_dim, bias = False))
        self.output_linear = nn.Linear(inner_dim, dim)
        
    def forward(self, q, k, v):
        f_q = self.input_linear(q).view(q.shape[0], q.shape[1], self.heads, -1).permute(2, 0, 1, 3)
        f_k = self.input_linear(k).view(k.shape[0], k.shape[1], self.heads, -1).permute(2, 0, 1, 3)
        f_v = self.input_linear(v).view(v.shape[0], v.shape[1], self.heads, -1).permute(2, 0, 1, 3)
        
        
        dots = cosine_distance(f_q, f_k.transpose(-1, -2)) # (h, q, n, 1)
        
        out = torch.matmul(dots, f_v)                      # (h, q, n, d_h)
        out = out.permute(1, 2, 0, 3).contiguous()
        out = out.view(out.shape[0], out.shape[1], -1)  # (q, n, d)
        
        return self.output_linear(out)


class CosineDistLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(CosineDistLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)      
        self.scale_factor = 2
        

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) 
        scores = self.scale_factor * (cos_dist)

        return scores


def cosine_distance(x1, x2):
    '''
    x1      =  [b, h, n, k]
    x2      =  [b, h, k, m]
    output  =  [b, h, n, m]
    '''
    dots = torch.matmul(x1, x2)

    norm_x1 = torch.norm(x1, dim=-1, keepdim=True)
    norm_x2 = torch.norm(x2, dim=-2, keepdim=True)

    scale = norm_x1 * norm_x2

    return (dots / scale)
