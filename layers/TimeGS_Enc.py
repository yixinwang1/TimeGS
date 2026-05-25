# from: https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
from functools import partial

import numpy as np
import torch
import torch.nn as nn

def get_norm_layer(kind='bn'):
    if not isinstance(kind, str):
        return kind
    if kind == 'bn':
        return nn.BatchNorm2d
    if kind == 'in':
        return nn.InstanceNorm2d
    if kind == 'gn':
        return nn.GroupNorm
    raise ValueError(f'Unknown norm block kind {kind}')


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=True, conv_kind='default',
                 dilation=1, in_dim=None, groups=1, second_dilation=None):
        super(ResnetBlock, self).__init__()
        self.dim = dim
        if second_dilation is None:
            second_dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout,
                                                conv_kind=conv_kind, dilation=dilation, in_dim=None, groups=groups,
                                                second_dilation=second_dilation)

        self.out_channnels = dim

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, conv_kind='default',
                         dilation=1, in_dim=None, groups=1, second_dilation=1):
        conv_layer = nn.Conv2d

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(dilation)]
        elif padding_type == 'zero':
            p = dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [conv_layer(dim, dim, kernel_size=3, padding=p, dilation=dilation),
                       norm_layer(dim) if not issubclass(norm_layer, nn.GroupNorm) else norm_layer(16, dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(second_dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(second_dilation)]
        elif padding_type == 'zero':
            p = second_dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv_layer(dim, dim, kernel_size=3, padding=p, dilation=second_dilation, groups=groups),
                       norm_layer(dim) if not issubclass(norm_layer, nn.GroupNorm) else norm_layer(16, dim),]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        before_x = x
        x = self.conv_block(x)
        out = x + before_x
        return out


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=4, n_downsampling=3, n_blocks=2, kernel_size=3, norm_layer=nn.BatchNorm2d,
                 padding_type='zero', conv_kind='default', activation=nn.ReLU(True),
                 use_skip=True,
                 up_norm_layer=nn.BatchNorm2d, affine=None,
                 up_activation=nn.ReLU(True), add_out_act=True,
                 max_features=1024, is_resblock_depthwise=False, 
                 dilation=1, second_dilation=None,):
        assert (n_blocks >= 0)
        super().__init__()
        self.use_skip = use_skip

        conv_layer = nn.Conv2d
        norm_layer = get_norm_layer(norm_layer)
        if affine is not None:
            norm_layer = partial(norm_layer, affine=affine)
        up_norm_layer = get_norm_layer(up_norm_layer)
        if affine is not None:
            up_norm_layer = partial(up_norm_layer, affine=affine)
        
        
        inp_module = [conv_layer(input_nc, ngf, kernel_size=kernel_size, padding=kernel_size//2),
                norm_layer(ngf) if not issubclass(norm_layer, nn.GroupNorm) else norm_layer(16, ngf),
                activation]
        self.inp_module = nn.Sequential(*inp_module)

        # downsample
        self.downs = nn.ModuleList([])
        for i in range(n_downsampling):
            mult = 2 ** i

            self.downs.append(nn.ModuleList([
                conv_layer(min(max_features, ngf * mult),
                                min(max_features, ngf * mult * 2),
                                kernel_size=3, stride=2, padding=1),
                norm_layer(min(max_features, ngf * mult * 2)) if not issubclass(norm_layer, nn.GroupNorm) else norm_layer(16, min(max_features, ngf * mult * 2)) ,
                activation
            ]))

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)


        # resnet blocks
        self.mid_res = nn.ModuleList([])
        for i in range(n_blocks):
            if is_resblock_depthwise:
                resblock_groups = feats_num_bottleneck
            else:
                resblock_groups = 1

            self.mid_res.append(ResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_kind=conv_kind, groups=resblock_groups,
                                    dilation=dilation, second_dilation=second_dilation))

        # upsample
        self.ups = nn.ModuleList([])
        self.skip_linear = nn.ModuleList([])
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)

            if self.use_skip and i>0:
                inp_shape = min(max_features, ngf * mult) * 2
            else:
                inp_shape = min(max_features, ngf * mult)
            out_shape = min(max_features, int(ngf * mult / 2))

            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(inp_shape, out_shape,
                                    kernel_size=3, stride=2, padding=1, output_padding=1),
                up_norm_layer(out_shape) if not issubclass(up_norm_layer, nn.GroupNorm) else up_norm_layer(16, out_shape), 
                up_activation
            ]))
            
            if self.use_skip and i>0:
                self.skip_linear.append(nn.Linear(out_shape*2, out_shape))

        out_module = [nn.Conv2d(ngf, output_nc, kernel_size=kernel_size, padding=kernel_size//2)]
        if add_out_act:
            out_module.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.out_module = nn.Sequential(*out_module)

    def forward(self, x):
        x = self.inp_module(x)

        res_fea = []
        for i, (conv, norm, act) in enumerate(self.downs):
            x = act(norm(conv(x)))
            
            if self.use_skip and i<(len(self.downs)-1):
                res_fea.append(x)
    
        for blk in self.mid_res:
            x = blk(x)

        for j, (conv, norm, act) in enumerate(self.ups):
            if self.use_skip and j>0:
                skip_x = res_fea.pop(-1)
                x = torch.cat([x, skip_x], dim=1)

            x = act(norm(conv(x)))
        
        x = self.out_module(x)

        return x

class TimeGSEncoder(nn.Module):
    def __init__(self, input_nc, gaussian, hidden_dim, conv_dim, len, ngf=4, n_downsampling=3, n_blocks=2, kernel_size=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.global_generator = GlobalGenerator(input_nc=input_nc, output_nc=conv_dim, ngf=ngf, n_downsampling=n_downsampling, n_blocks=n_blocks, kernel_size=kernel_size)
        self.project = nn.Linear(conv_dim * len, hidden_dim * gaussian)

    def forward(self, input):
        out = self.global_generator(input)      # [B*C, hidden, H, W]
        out = out.reshape(out.shape[0], -1)     # [B*C, hidden * len]
        out = self.project(out)                 # [B*C, hidden * gaussian]
        return out
