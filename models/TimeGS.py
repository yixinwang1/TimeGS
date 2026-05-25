import math
from termios import CWERASE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.TimeGS_Enc import TimeGSEncoder
from utils.mlp import MLP
from utils.sample import *
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import traceback

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.m = configs.m
        self.p = configs.p
        self.hidden_dim = configs.hidden_dim
        self.conv_dim = configs.conv_dim
        self.draft_len = configs.draft_len
        self.ratio = configs.ratio
        
        self.extend_len = configs.extend_len
        self.gaussian = self.pred_len // self.ratio + self.extend_len * 2

        period=[]
        for x in configs.period.split(','):
            pe = int(x)
            if pe == 0 or pe > self.pred_len + self.extend_len * 2 * self.ratio:
                pe = self.pred_len + self.extend_len * 2 * self.ratio
            period.append(pe)
        self.periods = torch.tensor(period)
        self.k = len(period)
        
        t = int(self.periods.min().item())
        self.m = min(self.m, (self.pred_len + t - 1) // t * 2 - 1)
        
        self.ori_width = 24
        image_height = configs.image_height
        image_width = configs.image_width
        self.image_size = (image_height, image_width)
        self.sample_len = image_height * image_width
        
        self.encoders = nn.ModuleList([TimeGSEncoder(1, self.gaussian, self.hidden_dim, self.conv_dim, self.sample_len, 
                                            configs.ngf, configs.n_downsampling, configs.n_blocks, configs.kernel_size) 
                                    for _ in range(self.k)])
        self.tem=0.1
        self.init_gaussian_dict(configs)

        self.v_mlps = nn.ModuleList([MLP(self.hidden_dim, self.p, [self.hidden_dim // 2]) for _ in range(self.k)])
        self.gaussian_mlps = nn.ModuleList([nn.Linear(self.hidden_dim, self.p * self.gaussian_dict.shape[0]) for _ in range(self.k)])

        self.feature_weight = nn.Parameter(torch.ones(configs.enc_in, self.k) / self.k)
        self.component_weight = nn.Parameter(torch.ones(configs.enc_in, self.p) / self.p)
        
        self.linear = nn.Linear(self.gaussian, self.pred_len)

        self.multi_gpu = configs.multi_gpu
        self.device = configs.device
        self.device_list = [f'cuda:{i}' for i in configs.device_list.split(',')]



    def init_gaussian_dict(self, configs):

        cholesky1 = torch.tensor([float(x) for x in configs.cholesky1.split(',')])
        cholesky2 = torch.tensor([float(x) for x in configs.cholesky2.split(',')])
        cholesky3 = torch.tensor([float(x) for x in configs.cholesky3.split(',')])
        coefficient = torch.tensor([float(x) for x in configs.coefficient.split(',')])
        
        ellipse_dict = torch.tensor(list(product(cholesky1, cholesky2, cholesky3, coefficient)))
        
        L11 = ellipse_dict[:, 0].clamp(min=1e-4).view(-1, 1, 1)
        L21 = ellipse_dict[:, 1].view(-1, 1, 1)
        L22 = ellipse_dict[:, 2].clamp(min=1e-4).view(-1, 1, 1)
        co = ellipse_dict[:, 3].view(-1, 1, 1)

        j_ = torch.arange(self.m) - self.m // 2
        grid = torch.arange(self.draft_len)
        j_ = j_.view(1, -1, 1)                                                                                  # [n, m, 1]
        grid = grid.view(1, 1, -1)                                                                              # [n, 1, d]

        # length, center point, left endpoint, and right endpoint of the interval
        center_float = self.draft_len // 2 - L21 * j_ / L11
        hi_float = torch.sqrt(1 - (L22 * j_).square().clamp(max=1)) / L11
        l_float = center_float - hi_float      
        r_float = center_float + hi_float
        valid = ((hi_float > 0)) & (grid >= l_float) & (grid <= r_float)                                        # [n, m, d]
        
        # Gaussian intensity
        dx = grid - center_float                                                                                # [n, m, d]
        exponent = - 0.5 * ((dx / L11).square() + ((j_ - L21 / L11 * dx) / L22).square())
        exponent = exponent * co
        alpha = exponent.exp()

        # only retain the valid parts
        alpha = valid * alpha                                                                                   # [n, m, d]

        alpha = alpha.reshape(-1, self.m * self.draft_len)
        
        # normalization
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        
        self.gaussian_dict = alpha


    def get_img(self, x):
        B, C, _ = x.shape
        
        # padding
        if self.seq_len % self.ori_width != 0:
            padding = (self.seq_len + self.ori_width - 1) // self.ori_width * self.ori_width - self.seq_len
            x = F.pad(x, pad=(padding, 0))
            
        # reshape
        x_img = x.reshape(B*C, 1, -1, self.ori_width).contiguous()
        
        # upsample
        x_img = sample2d(x_img, *self.image_size)

        return x_img

    
    def get_gaussian_parameter(self, x, x_img, idx):
        B, C, _ = x.shape

        # 2D Feature Extraction
        feat = self.encoders[idx](x_img)                                                                            # [B, C, g, hidden_dim]
        feat = feat.reshape(B, C, self.gaussian, self.hidden_dim)

        # Gaussian Kernel Generation
        v = self.v_mlps[idx](feat)                                                                                  # [B, C, g, p]

        gaussian_weight = self.gaussian_mlps[idx](feat)
        gaussian_weight = gaussian_weight.reshape(B, C, self.gaussian, self.p, self.gaussian_dict.shape[0])
        gaussian_weight = torch.softmax(gaussian_weight/self.tem, dim=-1)
        gaussian = gaussian_weight @ self.gaussian_dict.to(gaussian_weight.device)                                  # [B, C, g, p, m*draft_len]

        return v, gaussian

    # @torch.compile
    def move_gaussian_kernel(self, gaussian, period, pos):
        B, C, k = gaussian.shape[0], gaussian.shape[1], gaussian.shape[3]

        if self.m <= (self.pred_len + period - 1) // period * 2 - 1:
            m = self.m
        else:
            m = (self.pred_len + period - 1) // period * 2 - 1
            gaussian = gaussian[:, :, :, :, :, (self.m - m) // 2 : (self.m - m) // 2 + m, :]

        if self.draft_len >= period:
            le = self.draft_len // 2 - period // 2
            gaussian = gaussian[:, :, :, :, :, :, le:le + period]
        else:
            le = period // 2 - self.draft_len // 2
            gaussian = F.pad(gaussian, pad=(le, period - self.draft_len - le))

        gaussian = gaussian.reshape(B, C, self.gaussian, k, self.p, -1)                                             # [B, C, g, k, p, l]

        base_idx = torch.arange(self.pred_len, device=gaussian.device)
        base_idx = base_idx.view(1, 1, 1, 1, 1, -1)                                                                 # [1, 1, 1, 1, 1, L]
        shifted_idx = base_idx - pos + (m // 2 * period + period // 2) + 1                                          # [1, 1, g, 1, 1, L]
        shifted_idx = shifted_idx.expand(B, C, self.gaussian, k, self.p, self.pred_len)                             # [B, C, g, k, p, L]

        shifted_idx = torch.clamp(shifted_idx, 0, m * period + 1)
        gaussian = F.pad(gaussian, pad=(1, 1))

        gaussian = torch.gather(gaussian, dim=-1, index=shifted_idx)                                                # [B, C, g, k, p, L]
        
        pred = gaussian.sum(dim=2)                                                                                  # [B, C, k, p, L]

        return pred

    def gaussian_rasterize(self, v, gaussian):
        B, C = v.shape[0], v.shape[1]

        value = v.view(B, C, self.gaussian, self.k, self.p, 1, 1)                                                   # [B, C, g, k, p, 1, 1]
        gaussian = gaussian.view(B, C, self.gaussian, self.k, self.p, self.m, self.draft_len)                       # [B, C, g, k, p, m, d]
        gaussian = gaussian * value

        D_ = (torch.arange(self.gaussian, device=v.device) - self.extend_len) * self.ratio
        D_ = D_.view(1, 1, -1, 1, 1, 1)                                                                             # [1, 1, g, 1, 1, 1]
        D_ = D_ + torch.arange(self.k, device=v.device).view(1, 1, 1, -1, 1, 1) * self.ratio // self.k              # [1, 1, g, k, 1, 1]
        
        if torch.all(self.periods == self.periods[0]).item():
            period = self.periods[0]
            pred = self.move_gaussian_kernel(gaussian, period, D_)                                                  # [B, C, k, p, L]
        else:
            pred_list = []
            for i in range(self.k):
                period = self.periods[i]
                gaussian_ = gaussian[:, :, :, i:i+1, :, :, :]                                                       # [B, C, g, 1, p, m, d]
                pos = D_[:, :, :, i:i+1, :, :]                                                                      # [1, 1, g, 1, 1, 1]
                pred = self.move_gaussian_kernel(gaussian_, period, pos)
                pred_list.append(pred)
            pred = torch.concat(pred_list, dim=2)                                                                   # [B, C, k, p, L]

        return pred

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, debug=0):

        # RevIn
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev   

        x_enc = x_enc.permute(0, 2, 1)

        v_list = []
        gau_list = []
        
        for i in range(self.k):
            x_img = self.get_img(x_enc)
            v, gaussian = self.get_gaussian_parameter(x_enc, x_img, i)
            v_list.append(v)
            gau_list.append(gaussian)
        v = torch.stack(v_list, dim=-2)                                                             # [B, C, g, k, p]
        gau = torch.stack(gau_list, dim=-2)

        if self.multi_gpu:
            pred = self.gaussian_rasterize_parralel(v, gau)                                         # [B, C, k, p, L]
        else:
            pred = self.gaussian_rasterize(v, gau)                                                  # [B, C, k, p, L]
            
        fw = F.softmax(self.feature_weight / self.feature_weight.sum(dim=-1, keepdim=True), dim=-1)
        pred = pred * fw.view(1, self.enc_in, self.k, 1, 1)
        cw = F.softmax(self.component_weight / self.component_weight.sum(dim=-1, keepdim=True), dim=-1)
        pred = pred * cw.view(1, self.enc_in, 1, self.p, 1)
        
        pred = pred.sum(dim=(2, 3))                                                                 # [B, C, L]
        pred = pred.permute(0, 2, 1)

        pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return pred
    

    def gaussian_rasterize_parralel(self, v, gau):

        device_num = len(self.device_list)
        t = (v.shape[1] + device_num - 1) // device_num
        pred_list = [None] * device_num
        
        stream_list = [torch.cuda.Stream(device=self.device_list[i]) for i in range(device_num)]

        def _run(i):
            l, r = t * i, t * (i + 1)
            try:
                with torch.cuda.stream(stream_list[i]):
                    v_ = v[:, l:r, :, :, :].cpu().to(self.device_list[i], non_blocking=True)
                    gau_ = gau[:, l:r, :, :, :].cpu().to(self.device_list[i], non_blocking=True)

                    out = self.gaussian_rasterize(v_, gau_)
                    out = out.cpu().to(self.device, non_blocking=True)
                    pred_list[i] = out
            except Exception as e:
                print(f'\n!!! head {i} failed on {self.device_list[i]} !!!')
                traceback.print_exc()
                raise

        with ThreadPoolExecutor(max_workers=device_num) as pool:
            pool.map(_run, range(device_num))
        
        pred = torch.concat(pred_list, dim=1)                       # [B, C, k, p, L]
        return pred