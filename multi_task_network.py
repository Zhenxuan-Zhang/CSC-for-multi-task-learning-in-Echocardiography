import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import math
    
class CoPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x,x1, **kwargs):
        return self.fn(self.norm(x),self.norm(x1), **kwargs)

class CoAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x,x1):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        qkv1 = self.to_qkv(x1).chunk(3, dim = -1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv1)
        dots = einsum('b h i d, b h j d -> b h i j', q, k1) * self.scale

        attn = dots.softmax(dim=-1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class CoTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CoPreNorm(dim, CoAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))]))

    def forward(self, x,x1):
        for attn, ff in self.layers:
            x = attn(x,x1) + x
            x = ff(x) + x
        return self.norm(x)
    
class cosa(nn.Module):
    def __init__(self, image_size, patch_size, in_channels ,num_frames, depth = 4, heads = 3,num_classes=1, pool = 'cls', 
                 dim = 8, dim_head = 64, dropout = 0.,emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b (t c) (h w) (p1 p2)', p1 = patch_size, p2 = patch_size),
        )
        self.pos_embedding = PositionalEmbedding(num_patches*patch_size**2)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = CoTransformer(patch_size**2, depth, heads, dim_head, dim*scale_dim, dropout)
        self.img_out = image_size
        self.time = num_frames
        self.channel = in_channels
        self.patch_size = patch_size
    def forward(self, x1,x2):
        x1,x2 = rearrange(x1, 'b c t n d -> b t c n d'), rearrange(x2, 'b c t n d -> b t c n d')
        x1,x2 = self.to_patch_embedding(x1),self.to_patch_embedding(x2)
        b, t, n, d = x1.shape
        pos1,pos2 = self.pos_embedding(x1,n,d),self.pos_embedding(x2,n,d)
        x1 = x1 + pos1
        x2 = x2 + pos2
        x1 = rearrange(x1, 'b t n d -> (b t) n d')
        x2 = rearrange(x2, 'b t n d -> (b t) n d')
        x = self.transformer(x1,x2)
        patch_height = int(self.img_out/self.patch_size)
        x = rearrange(x, '(t c) (ph pw) (p1 p2) -> t c (ph p1) (pw p2)', t = self.time, c = self.channel,
                      ph = patch_height, p1 = self.patch_size)
        x = x.unsqueeze(0)
        x = rearrange(x, 'b t c n d -> b c t n d')
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x_attn = attn(x) 
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x),self.norm(x_attn)

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb
        self.mlp = nn.Sequential(nn.LayerNorm(self.demb),nn.Linear(self.demb, 1))
        inv_freq = 1 / (10 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq,imgsize,imgdim, bsz=None):
        pos_seq = rearrange(pos_seq, 'b t n d -> b t (n d)')
        pos_seq = self.mlp(pos_seq)
        pos_seq = pos_seq.squeeze(0)
        pos_seq = pos_seq.squeeze(1)
        #print(pos_seq)
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        if bsz is not None:
            pos_emb = pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            pos_emb = pos_emb[:,None,:]
        #print(pos_emb.shape)
        pos_emb = rearrange(pos_emb, 't b (n d) -> b t n d',n=imgsize,d=imgdim)
        return pos_emb
    
class STFH(nn.Module):
    def __init__(self, image_size,patch_size, in_channels ,num_frames, depth = 4, heads = 3,num_classes=1, pool = 'cls', 
                 dim = 8, dim_head = 64, dropout = 0.,emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b (t c) (h w) (p1 p2)', p1 = patch_size, p2 = patch_size),
        )
        self.pos_embedding = PositionalEmbedding(num_patches*patch_size**2)
        self.dropout = nn.Dropout(emb_dropout)
        self.spatio_temporal_transformer = Transformer(patch_size**2, depth, heads, dim_head, dim*scale_dim, dropout)
        self.img_out = image_size
        self.time = num_frames
        self.channel = in_channels
        self.patch_size = patch_size
        
       
    def forward(self, x):
        #print('before p_emb',x.shape)
        theta = 5
        pool = nn.MaxPool3d(kernel_size=theta, stride=1, padding=(theta - 1) // 2)
        x_pool = pool(x)
        x = 2*x - x_pool
        x = rearrange(x, 'b c t n d -> b t c n d')
        x = self.to_patch_embedding(x)
        b, t, n, d = x.shape
        pos = self.pos_embedding(x,n,d)
        x = x + pos
        x = self.dropout(x)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x,x_att = self.spatio_temporal_transformer(x)
        patch_height = int(self.img_out/self.patch_size)
        x = rearrange(x, '(t c) (ph pw) (p1 p2) -> t c (ph p1) (pw p2)', t = self.time, c = self.channel,
                      ph = patch_height, p1 = self.patch_size)
        x = x.unsqueeze(0)
        x = rearrange(x, 'b t c n d -> b c t n d')
        x_att = rearrange(x_att, '(t c) (ph pw) (p1 p2) -> t c (ph p1) (pw p2)', t = self.time, c = self.channel,
                      ph = patch_height, p1 = self.patch_size)
        x_att = x_att.unsqueeze(0)
        x_att = rearrange(x_att, 'b t c n d -> b c t n d')
        return x,x_att
    
class Dual_path(nn.Module):
    def __init__(self, image_size, in_channels ,num_frames, depth = 4, heads = 3,num_classes=1, pool = 'cls', 
                 dim = 8, dim_head = 64, dropout = 0.,emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        #assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        #num_patches = (image_size // patch_size) ** 2
        #patch_dim = in_channels * patch_size ** 2
        self.spatio_stfh = STFH(image_size=image_size, patch_size=image_size//2, in_channels=in_channels ,num_frames=num_frames)
        self.temporal_stfh = STFH(image_size=image_size, patch_size=image_size, in_channels=in_channels ,num_frames=num_frames)
       
    def forward(self, x):
        #print('before p_emb',x.shape)
        theta = 5
        pool = nn.MaxPool3d(kernel_size=theta, stride=1, padding=(theta - 1) // 2)
        x_pool = pool(x)
        x_bound = 2*x - x_pool
        x_bound,x_att_b = self.spatio_stfh(x_bound)
        x,x_att_t = self.temporal_stfh(x)
        x_croatt = x_att_t*x_att_b
        x = x+x_bound+x_croatt
        return x

class TimeDistributed(nn.Module):
    def __init__(self, layer, time_steps):        
        super(TimeDistributed, self).__init__()
        self.layers = nn.ModuleList([layer for i in range(time_steps)])

    def forward(self, x):
        x = rearrange(x, 'b c t n d -> b t c n d')
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([]).cuda()
        #output = torch.tensor([])
        for i in range(time_steps):
            output_t = self.layers[i](x[:, i, :, :, :])
            output_t  = output_t.unsqueeze(1)
            output = torch.cat((output, output_t ), 1)
        output = rearrange(output, 'b t c n d -> b c t n d')
        return output


class EncoderBottleneck3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=[1,2,2], bias=False),
            nn.BatchNorm3d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm3d(width)

        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=[1,2,2], groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm3d(width)

        self.conv3 = nn.Conv3d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x_down = self.downsample(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim,seq_frame):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=7, stride=[1,2,2], padding=3, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.encoder1 = EncoderBottleneck3d(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck3d(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck3d(out_channels * 4, out_channels * 8, stride=2)

        self.conv2 = nn.Conv3d(out_channels * 8, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(out_channels * 4)

    def forward(self, x):
        x = rearrange(x, 'b t c n d -> b c t n d')
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x, x1, x2, x3

class DecoderBottleneck3d(nn.Module):
    def __init__(self, in_channels, out_channels,seq_frame, scale_factor=2):
        super().__init__()

        self.upsample = TimeDistributed(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True), time_steps = seq_frame)
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False))

    def forward(self, x, x_concat=None):
        
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
        x = self.layer(x)
        return x

class dconv3d(nn.Module):
    def __init__(self, in_channels, out_channels,d_rate, stride=1, base_width=64):
        super().__init__()

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm3d(width)

        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=[1,1,1], groups=1, padding=d_rate, dilation=d_rate, bias=False)
        self.norm2 = nn.BatchNorm3d(width)

        self.conv3 = nn.Conv3d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        return x
    
class task_specific_decoder(nn.Module):
    def __init__(self, out_channels, class_num, drop_rate,seq_frame):
        super().__init__()
        self.d_rate = drop_rate
        self.decoder1 = DecoderBottleneck3d(out_channels * 8, out_channels * 2,seq_frame)
        self.decoder2 = DecoderBottleneck3d(out_channels * 4, out_channels,seq_frame)
        self.dropout = TimeDistributed(torch.nn.Dropout(p=self.d_rate),time_steps = seq_frame)
        
    def forward(self, x, x2, x3):
        x = self.decoder1(x, x3)
        x = self.dropout(x)
        x = self.decoder2(x, x2)
        #x = rearrange(x, 'b c t n d -> b t c n d')
        return x

    
class segment_head(nn.Module):
    def __init__(self, out_channels, class_num, drop_rate,seq_frame):
        super().__init__()
        self.decoder3 = DecoderBottleneck3d(out_channels * 2, int(out_channels * 1 / 2),seq_frame)
        self.decoder4 = DecoderBottleneck3d(int(out_channels * 1 / 2), int(out_channels * 1 / 8),seq_frame)
        self.conv1 = nn.Conv3d(int(out_channels * 1 / 8), class_num, kernel_size=1)
        
    def forward(self, x, x1):
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)
        x = rearrange(x, 'b c t n d -> b t c n d')
        return x

class point_head(nn.Module):
    def __init__(self, out_channels, class_num, seq_frame,height,weight):
        super().__init__()
        self.seq_frame = seq_frame
        self.height = height
        self.weight = weight
        self.decoder3 = DecoderBottleneck3d(out_channels * 2, int(out_channels * 1 / 2),seq_frame)
        self.decoder4 = DecoderBottleneck3d(int(out_channels * 1 / 2), int(out_channels * 1 / 8),seq_frame)
        self.conv1 = nn.Conv3d(int(out_channels * 1 / 8), class_num, kernel_size=1)
        self.conv2 = nn.Conv3d(class_num,1, kernel_size=1)
        self.mlp_out = nn.Sequential(
            nn.LayerNorm(self.height*self.weight),
            nn.Linear(self.height*self.weight, 2))
        
    def forward(self, x, x1):
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)
        x_map = self.conv2(x)
        x_map = rearrange(x_map, 'b c t n d -> b t c n d')
        x = rearrange(x, 'b c t n d -> b t c n d')
        x_pot = rearrange(x, 'b t c n d -> b t c (n d)')
        x_pot = self.mlp_out(x_pot)
        return x_pot,x_map
    
class frame_head(nn.Module):
    def __init__(self, out_channels, class_num,seq_frame):
        super().__init__()
        self.out_ch = out_channels
        self.class_num = class_num
        self.seq_frame = seq_frame
        self.dconv3d2 = dconv3d(self.out_ch*1, self.out_ch*2, d_rate = 3, stride=2)
        self.avgpool2d2  = TimeDistributed(nn.AdaptiveAvgPool2d(8), time_steps = seq_frame)
        self.dconv3d3 = dconv3d(self.out_ch*2, self.out_ch*4, d_rate = 2, stride=2)
        self.avgpool2d3  = TimeDistributed(nn.AdaptiveAvgPool2d(2), time_steps = seq_frame)
        self.dconv3d4 = dconv3d(self.out_ch*4, self.out_ch*8, d_rate = 1, stride=2)
        self.avgpool2d4  = TimeDistributed(nn.AdaptiveAvgPool2d(1), time_steps = seq_frame)
        self.mlp_out = nn.Sequential(
            nn.LayerNorm(self.out_ch*8),
            nn.Linear(self.out_ch*8, self.class_num))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self,x):
        #print(x.shape)
        x = self.dconv3d2(x)
        #print(x.shape)
        x = self.avgpool2d2(x)
        x = self.dconv3d3(x)
        x = self.avgpool2d3(x)
        x = self.dconv3d4(x)
        x = self.avgpool2d4(x)
        #print(x.shape)
        x = rearrange(x, 'b t c n d -> (c n d)(b t)')
        x_reg = self.mlp_out(x)
        x_reg.squeeze()
        #x_reg = rearrange(x_reg, 'n s -> s n')
        return x_reg
    
class TaskAttention(nn.Module):
    def __init__(self, in_ch, ratio=16):
        super(TaskAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        self.fc = nn.Sequential(nn.Conv3d(in_ch*2, in_ch//16, kernel_size=1),
                               nn.ReLU(),
                               nn.Conv3d(in_ch // 16, in_ch*2, kernel_size=1))
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv3d(in_ch*2, in_ch, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm3d(in_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x1,x2):
        x = torch.cat([x1, x2], dim=1)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        x = self.sigmoid(out)+x
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x
    
class pattern_strcture(nn.Module):
    def __init__(self, image_size, in_channels ,num_frames, depth = 4, heads = 3,num_classes=1, pool = 'cls', 
                 dim = 8, dim_head = 64, dropout = 0.,emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        self.spatio_stfh = STFH(image_size=image_size, patch_size=image_size//4, in_channels=in_channels ,num_frames=num_frames)
        self.ta = TaskAttention(in_channels)
        size = 32
        self.cosa = cosa(image_size=size,patch_size=int(size//2), in_channels=32 ,num_frames=30)
       
    def forward(self,x1,x2,x3):
        x1_sa,x_att_b1 = self.spatio_stfh(x1)
        x2_sa,x_att_b2 = self.spatio_stfh(x2)
        x3_sa,x_att_b3 = self.spatio_stfh(x3)
        beta = 0.1
        x1 = self.cosa(beta*self.ta(x2_sa,x3_sa),(1-beta)*x1_sa)
        x2 = self.cosa(beta*self.ta(x1_sa,x3_sa),(1-beta)*x2_sa)
        x3 = self.cosa(beta*self.ta(x2_sa,x1_sa),(1-beta)*x3_sa)
        return x1,x2,x3

seq_frame = 30

class Mtl_net(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num, drop_rate,seq_frame
                 ,mode,height,weight):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim,seq_frame)
        self.seg_decoder = task_specific_decoder(out_channels, class_num, drop_rate,seq_frame)
        self.pot_decoder = task_specific_decoder(out_channels, class_num, drop_rate,seq_frame)
        self.frm_decoder = task_specific_decoder(out_channels, class_num, drop_rate,seq_frame)
        self.seg_head = segment_head(out_channels, class_num, drop_rate,seq_frame)
        self.pot_head = point_head(out_channels, class_num=4,seq_frame=seq_frame,height=height,weight=weight)
        self.frm_head = frame_head(out_channels, class_num=3,seq_frame=seq_frame)
        self.mode = mode
        self.stps = pattern_strcture(image_size=int(img_dim/4), in_channels=out_channels ,num_frames=seq_frame)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        if self.mode == 'seg':
            x_seg = self.seg_decoder(x, x2, x3)
            x_seg = self.seg_head(x_seg,x1)
            return x_seg
        elif self.mode == 'pot':
            x_pot = self.pot_decoder(x, x2, x3)
            x_pot = self.pot_head(x_pot,x1)
            return x_pot
        elif self.mode == 'frm':
            x_frm = self.frm_decoder(x, x2, x3)
            x_frm = self.frm_head(x_frm)
            return x_frm
        elif self.mode == 'mtl':
            x_seg = self.seg_decoder(x, x2, x3)
            x_pot = self.pot_decoder(x, x2, x3)
            x_frm = self.frm_decoder(x, x2, x3)
            x_seg,x_pot,x_frm = self.stps(x_seg,x_pot,x_frm)
            x_seg = self.seg_head(x_seg,x1)
            x_pot,x_potmap = self.pot_head(x_pot,x1)
            x_frm = self.frm_head(x_frm)
            return x_pot,x_potmap,x_frm,x_seg  #x_pot=(1,30,4,2),x_frm(30,3),x_seg(1,30,1,128,128)