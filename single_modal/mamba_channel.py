import math
# from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Spe_SSM(nn.Module):
    def __init__(
            self,
            imagesize,
            dim,
            d_conv=3,  # 15
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.imagesize = imagesize  # 96  224  s*s*c
        self.dim = dim // 4  # change_name
        self.d_conv = d_conv
        # self.in_proj = nn.Linear(dim, self.dim * 2, bias=bias, **factory_kwargs)
        self.in_proj = nn.Sequential(
            nn.Linear(dim, self.dim * 2, bias=bias),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim * 2, bias=bias),
            nn.LayerNorm(self.dim * 2),
        )

        # x_in:b c l -> b l c, along c conv, compress the neighbor sequence
        self.conv2d = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            groups=self.dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # Cnn+act+residual from gate mlp, structure strategy
        self.act = nn.SiLU()

        K = 2
        self.K = K
        zip1 = 8  # 8
        zip2 = 4
        # zip3 = 2
        self.to_B = nn.Sequential(
            Rearrange('B C (h N_h) (w N_w) -> B N_h N_w (h w C)', h=zip1, w=zip1),
            nn.LayerNorm(zip1 * zip1 * K * self.dim),
            nn.Linear(zip1 * zip1 * K * self.dim, K * self.dim),
            nn.LayerNorm(K * self.dim),
            Rearrange('B (h N_h) (w N_w)  C-> B N_h N_w (h w C)', h=zip2, w=zip2),
            nn.LayerNorm(zip2 * zip2 * K * self.dim),
            nn.Linear(zip2 * zip2 * K * self.dim, K * self.dim),
            nn.LayerNorm(K * self.dim),
            Rearrange('B H W C -> B C H W')
        )

        self.to_C = nn.Sequential(
            Rearrange('B C (h N_h) (w N_w) -> B N_h N_w (h w C)', h=zip1, w=zip1),
            nn.LayerNorm(zip1 * zip1 * K * self.dim),
            nn.Linear(zip1 * zip1 * K * self.dim, K * self.dim),
            nn.LayerNorm(K * self.dim),
            Rearrange('B (h N_h) (w N_w)  C-> B N_h N_w (h w C)', h=zip2, w=zip2),
            nn.LayerNorm(zip2 * zip2 * K * self.dim),
            nn.Linear(zip2 * zip2 * K * self.dim, K * self.dim),
            nn.LayerNorm(K * self.dim),
            Rearrange('B H W C -> B C H W')
        )

        self.token_height = 8
        self.token_width = 8
        patch_num_x = self.imagesize // self.token_height
        patch_num_y = self.imagesize // self.token_width
        token_dim = self.token_height * self.token_width * self.dim * self.K
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_num_x, patch_num_y, self.K * self.dim))
        ########################################################################################
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=self.token_height, p2=self.token_width),
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, self.K * self.dim),
            nn.LayerNorm(self.K * self.dim))
        ########################################################################################

        # self.to_dt = nn.Sequential(  # 205
        #     nn.Linear(K * self.dim, 2 * K * self.dim),  # ratio=2,k=2
        #     nn.Linear(2 * K * self.dim, K * self.dim),
        #     nn.LayerNorm(self.dim * K),
        #     Rearrange('B H W C -> B C H W')
        # )

        self.to_dt = nn.Sequential(  # 243
            Rearrange('B H W C -> B C (H W)'),
            nn.Linear(patch_num_x * patch_num_y, patch_num_x * patch_num_y // 8),  # ratio=2,k=2
            nn.Linear(patch_num_x * patch_num_y // 8, patch_num_x * patch_num_y),
            nn.LayerNorm(patch_num_x * patch_num_y),
        )

        # ht = Aht + Bx, model hope hang history by A
        self.A_logs = self.A_log_init((self.imagesize // 32) * (self.imagesize // 32),
                                      (self.imagesize // self.token_height) * (self.imagesize // self.token_width),
                                      copies=K, merge=True)  # (K=2, D, N)
        self.lnA = nn.LayerNorm(int((self.imagesize // 32) * (self.imagesize // 32)))

        # the residual project matrix of x_in
        self.Ds = self.D_init((self.imagesize // self.token_height) * (self.imagesize // self.token_width), copies=K,
                              merge=True)  # (K=2, D, N)
        # mamba_core
        self.selective_scan = selective_scan_fn

        self.gatefusion = nn.Sequential(
            nn.Linear(2 * (self.imagesize // self.token_height) * (self.imagesize // self.token_width), 2, bias=False),
            nn.Softmax(dim=-1)
        )

        # norm the output of SS2D
        self.out_norm = nn.LayerNorm((self.imagesize // self.token_height) * (self.imagesize // self.token_width))

        self.to_out = nn.Sequential(
            nn.Linear(self.dim, self.token_height * self.token_width * self.dim),
            nn.Dropout(dropout),
        )

        # proj the output gate attention
        # self.out_proj = nn.Linear(self.dim, dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Sequential(
            nn.Linear(self.dim, dim * 2, bias=bias),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=bias),
            nn.LayerNorm(dim),
        )

        # self.out_proj = nn.Conv2d(self.dim, self.dim, 3, 1, 1, groups=dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.proj_drop = nn.Dropout(dropout)

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization? HiPPO? is not match
        # [1,2,,3,4,...,16]
        # row = d_inner
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        # log
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        # [1,1,1,1,1] equal norm residual
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, H, W, C = x.shape  # 输入影像尺寸
        L = C  # 将spe当作token，将不同spe排列视为token排序，spa值作为编码值，但是spa信息在第二维（transformer中token是才是第二维）

        K = 2
        x_f = x.reshape(B, H * W, C)
        x_fb = torch.stack([x_f, torch.flip(x_f, dims=[-1])], dim=1).view(B, 2, -1, L)
        # x_fb = x_f.view(B, 1, -1, L)
        x_fb = x_fb.permute(0, 2, 1, 3)
        x_fb = rearrange(x_fb, 'b (h w) k c -> b (k c) h w', h=H)

        Bs = self.to_B(x_fb)  # b (k c) h/32, w/32
        Cs = self.to_C(x_fb)  # b (k c) h/32 w/32

        # dts = self.to_dt(x_fb)  # b (k c) h w

        x_fb_in = self.to_patch_embedding(x_fb)
        b, h, w, _ = x_fb_in.shape
        x_fb = self.proj_drop(x_fb_in)
        x_fb += self.pos_embedding[:, :h, :w]

        # dts = self.to_dt(x_fb)  # 205
        dts = rearrange(self.to_dt(x_fb), 'b c (h w) -> b c h w', h=h)  # 243

        # xs = rearrange(x_fb, 'b (k c) h w -> b (k h w) c', k=K).float()  # (b, k * h * w, l)
        xs = rearrange(x_fb, 'b h w (k c) -> b (k h w) c', k=K).float()  # (b, k * h * w, l)

        dts = rearrange(dts.contiguous(), 'b (k c) h w -> b (k h w) c', k=K).float()  # (b, k * h * w, l)
        Bs = rearrange(Bs, 'b (k c) h w -> b k (h w) c', k=K).float()  # (b, k, h * w/(32*32), l)
        Cs = rearrange(Cs, 'b (k c) h w -> b k (h w) c', k=K).float()  # (b, k, h * w/(32*32), l)
        Ds = self.Ds.float().view(-1)  # (k * h * w)
        # A=-[1,2,3,...,16]
        As = -torch.exp(self.A_logs.float()).view(-1, int(self.imagesize / 32) * int(
            self.imagesize / 32))  # (k * h * w, d_state)
        # As = self.lnA(As)
        # ht=A*ht+B*xs:b k*d k d_state
        # y = C*ht+D*xt: b k*d l
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # 恢形状
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, 1, -1, L)  # 后两个顺序变正常

        return out_y[:, 0], inv_y[:, 0]  # forward and backwardout_y[:, 0]  #

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape  # 输入影像尺寸
        ##########
        xz = self.in_proj(x)  # 首先利用线性变换将原始维度转换为mamba内部维度 256 256 28 =>256 256 14
        # 2*256*256=>2*
        x, z = xz.chunk(2, dim=-1)  # (b, H, W, C) 将x分割成两块,一块留下备用

        x = self.act(self.conv2d(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)  # (b, d, h, w)#exploit the shallow feature
        y1, y2 = self.forward_core(x)  # 进行扫描
        assert y1.dtype == torch.float32
        # add fusion
        # y = y1 + y2  # 不同扫描映射的结果相加

        # gatefusion
        # y: b hw c -> b hw fusion a sequence
        mean_y1 = torch.mean(y1, dim=-1)
        mean_y2 = torch.mean(y2, dim=-1)
        # gate = b 2*hw -> b k, where k=2
        gate = torch.cat([mean_y1, mean_y2], dim=1)
        gate = self.gatefusion(gate)
        # gate = b k 1
        gate = gate.unsqueeze(-1)
        # b 1 1 * b hw c
        y = gate[:, 0:1] * y1 + gate[:, 1:2] * y2

        # y = y1
        # y->b c hw
        y = torch.transpose(y, dim0=1, dim1=2).contiguous()  # .view(B, C, -1)  # 将h*w展开
        # 输出头
        y = self.out_norm(y)  # 层归一化
        y = rearrange(y, 'b c (h w)->b h w c', h=(self.imagesize // self.token_height))

        # recover_spa
        y = self.to_out(y)
        y = rearrange(y, 'b h w (p1 p2 c)->b (h p1) (w p2) c', p1=self.token_height, p2=self.token_width, c=self.dim)
        y = y * F.silu(z)  # 留下的另一块形成权重进行加权, the block can change to mask

        out = self.out_proj(y)  # 恢复成原来维度

        if self.dropout is not None:
            out = self.dropout(out)
        return out


class B_Gen(nn.Module):
    def __init__(self, K=4, dinner=64):
        super(B_Gen, self).__init__()
        self.res_pos = 2
        self.d_inner = dinner
        self.K = K
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.d_inner * K, out_channels=self.d_inner * K, kernel_size=3, padding=1),
        #     nn.SiLU(),
        #     nn.Conv1d(in_channels=self.d_inner * K, out_channels=self.d_inner * K, kernel_size=3, padding=1),
        # )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.d_inner * K, out_channels=self.d_inner * K, kernel_size=5, padding=2,
                      groups=self.d_inner * K),
            nn.Conv2d(in_channels=self.d_inner * K, out_channels=self.d_inner * K, kernel_size=7, padding=9, dilation=3,
                      groups=self.d_inner * K),
            nn.BatchNorm2d(self.d_inner * K),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 2, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(2, 1, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            Rearrange('B d c H W -> B (H W) (d c)')
        )
        self.linear_layer1 = nn.Sequential(
            nn.Linear(self.d_inner * K, self.d_inner * K // 2, bias=True),
            nn.ReLU(),
            nn.Linear(self.d_inner * K // 2, self.d_inner * K, bias=True),
        )

        self.linear_layer2 = nn.Sequential(
            nn.Linear(self.d_inner * K, self.d_inner * K // 2, bias=True),
            nn.ReLU(),  # yangtiaohanshu, no stake should two py,c,2c,4c,2c,c, use kan yangtiaohanshu
            nn.Linear(self.d_inner * K // 2, self.d_inner * K, bias=True),
        )

        self.linear_layer3 = nn.Sequential(
            nn.Linear(self.d_inner * K, self.d_inner * K // 2, bias=True),
            nn.ReLU(),
            nn.Linear(self.d_inner * K // 2, self.d_inner * K, bias=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape  # 输入影像尺寸
        # 1D
        x = self.conv2d(x)
        # 3D
        x = rearrange(x, 'b (d c) h w->b d c h w', d=1)
        x = self.conv1(x)
        x = self.linear_layer1(x) + x
        x = self.linear_layer2(x) + x
        x = self.linear_layer3(x) + x
        x = rearrange(x, 'b l (k c)->b k c l', k=self.K)
        return x


class Spa_SSM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=4,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  #
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.in_proj_rgb = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.conv2d_rgb = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act_rgb = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state, bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.Bs_proj = (
            nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
        )
        self.Bs_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.Bs_proj], dim=0))  # (K=4, N, inner)
        del self.Bs_proj

        self.to_B = B_Gen(K=4, dinner=self.d_inner)

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )  # 偏置尺寸等于原始影像
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        #
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.gatefusion = nn.Sequential(
            nn.Linear(4 * self.d_inner, 4, bias=False),
            nn.Softmax(dim=-1)
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):  # , rgb: torch.Tensor
        B, C, H, W = x.shape  # 输入影像尺寸
        L = H * W  # 将空间当作token，将不同空间排列视为token排序，光谱值作为编码值，但是光谱信息在第二维（transformer中token是才是第二维）
        K = 4  # 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  # 两个不同排列，左右上下，上下左右
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l) # 形成四个排列，左右上下，上下左右，右左下上，下上右左
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L),
                             self.x_proj_weight)  # 四个序列，利用线性变换进行四个不同的映射，注意此时光谱（第三个维度）变成dt+2*state

        dts, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state], dim=2)  # 映射的特征延光谱维进行分割生成dt，B，C
        # dts = x_dbl
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)  # dt视为偏置

        Bs = self.to_B(rearrange(xs, 'b k c (h w)->b (k c) h w', h=H, w=W))
        Bs = torch.einsum("b k d l, k c d -> b k c l", Bs.view(B, K, -1, L),
                          self.Bs_proj_weight)  # 四个序列，利用线性变换进行四个不同的映射，注意此时光谱（第三个维度）变成dt+2*state

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # ht=A*ht+B*xs:b k*d k d_state
        # y = C*ht+D*xt: b k*d l
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # 恢复复形状
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)  # 后两个顺序变正常
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)  # 第二个顺序变回H W
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1,
                                                                                                   L)  # 最后一个变回 H W

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y  # 左右上下，右左下上，上下左右，下上右左

    def forward(self, x: torch.Tensor, **kwargs):  # , rgb: torch.Tensor
        B, H, W, C = x.shape  # 输入影像尺寸

        xz = self.in_proj(x)  # 首先利用线性变换将原始维度转换为mamba内部维度
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) 将x分割成两块一块留下备用
        del xz
        torch.cuda.empty_cache()

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)#扫描前预处理

        y1, y2, y3, y4 = self.forward_core(x)  # 进行扫描, rgb

        assert y1.dtype == torch.float32
        # gatefusion
        mean_y1 = torch.mean(y1, dim=-1)
        mean_y2 = torch.mean(y2, dim=-1)
        mean_y3 = torch.mean(y3, dim=-1)
        mean_y4 = torch.mean(y4, dim=-1)
        gate = torch.cat([mean_y1, mean_y2, mean_y3, mean_y4], dim=1)  #

        del mean_y1, mean_y2, mean_y3, mean_y4
        torch.cuda.empty_cache()

        gate = self.gatefusion(gate)
        gate = gate.unsqueeze(-1)
        y = gate[:, 0:1] * y1 + gate[:, 1:2] * y2 + gate[:, 2:3] * y3 + gate[:, 3:4] * y4

        del y1, y2, y3, y4
        torch.cuda.empty_cache()

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # 将h*w展开
        y = self.out_norm(y)  # 层归一化
        y = y * F.silu(z)  # 留下的另一块形成权重进行加权

        out = self.out_proj(y)  # 恢复成原来维度
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class B_Gen1(nn.Module):
    def __init__(self, K=1, dinner=64):
        super(B_Gen1, self).__init__()
        self.res_pos = 2
        self.d_inner = dinner
        self.K = K

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.d_inner * K, out_channels=self.d_inner * K, kernel_size=5, padding=2,
                      groups=self.d_inner * K),
            nn.Conv2d(in_channels=self.d_inner * K, out_channels=self.d_inner * K, kernel_size=7, padding=9, dilation=3,
                      groups=self.d_inner * K),
            nn.BatchNorm2d(self.d_inner * K),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 2, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(2, 1, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            Rearrange('B d c H W -> B (H W) (d c)')
        )

        self.linear_layer1 = nn.Sequential(
            nn.Linear(self.d_inner * K, self.d_inner * K // 2, bias=True),
            nn.ReLU(),
            nn.Linear(self.d_inner * K // 2, self.d_inner * K, bias=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape  # 输入影像尺寸
        x1 = self.conv2d(x) + x
        # 3D
        x1 = rearrange(x1, 'b (d c) h w->b d c h w', d=1)
        x2 = self.conv1(x1) + rearrange(x.permute(0, 2, 3, 1), 'b h w c->b (h w) c') + \
             rearrange(x1, 'b d c h w->b (h w) (d c)')

        x = self.linear_layer1(x2) + x2 + rearrange(x1, 'b d c h w->b (h w) (d c)')
        x = rearrange(x, 'b l (k c)->b k c l', k=self.K)
        return x


class B_Gen2(nn.Module):
    def __init__(self, K=1, dinner=64):
        super(B_Gen2, self).__init__()
        self.res_pos = 2
        self.d_inner = dinner
        self.K = K

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.d_inner * K, out_channels=self.d_inner * K, kernel_size=5, padding=2,
                      groups=self.d_inner * K),
            nn.Conv2d(in_channels=self.d_inner * K, out_channels=self.d_inner * K, kernel_size=7, padding=9, dilation=3,
                      groups=self.d_inner * K),
            nn.BatchNorm2d(self.d_inner * K),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 2, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(2, 1, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 2, (7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(2, 1, (7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(1, 2, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(2, 1, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            Rearrange('B d c H W -> B (H W) (d c)')
        )

        # self.linear_layer1 = nn.Sequential(
        #     nn.Linear(self.d_inner * K, self.d_inner * K // 2, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(self.d_inner * K // 2, self.d_inner * K, bias=True),
        # )

    def forward(self, x):
        B, C, H, W = x.shape  # 输入影像尺寸
        x1 = self.conv2d(x) + x
        # x1 = x
        # 3D
        x1 = rearrange(x1, 'b (d c) h w->b d c h w', d=1)
        x2 = self.conv1(x1) + x1 + rearrange(x, 'b (d c) h w->b d c h w', d=1)
        x21 = self.conv2(x2) + x2 + x1
        x22 = self.conv3(x21) + rearrange(x21, 'b d c h w->b (h w) (d c)') + rearrange(x2, 'b d c h w->b (h w) (d c)')

        # x = self.linear_layer1(x22) + x22 + rearrange(x21, 'b d c h w->b (h w) (d c)')
        # x = rearrange(x, 'b l (k c)->b k c l', k=self.K)
        x = rearrange(x22, 'b (h w) c->b c h w', h=H, w=W)
        return x


class B_Gen3(nn.Module):
    def __init__(self, K=1, dinner=64):
        super(B_Gen3, self).__init__()
        self.K = K
        self.B_Gen1 = B_Gen2(K=K, dinner=dinner)
        # self.B_Gen2 = B_Gen2(K=K, dinner=dinner)
        # self.B_Gen3 = B_Gen2(K=K, dinner=dinner)

    def forward(self, x):
        B, C, H, W = x.shape  # 输入影像尺寸
        x = self.B_Gen1(x) + x
        # x = self.B_Gen2(x) + x
        # x = self.B_Gen3(x) + x
        x = rearrange(x, 'b (k c) h w->b k c (h w)', k=self.K)
        return x


class Spa_SSM1(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=4,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  #
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # self.in_proj_rgb = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.to_dt = B_Gen3(K=1, dinner=self.d_inner)
        self.dt_proj = (
            nn.Linear(self.d_inner, self.dt_rank, bias=False, **factory_kwargs),
        )
        self.dt_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_proj], dim=0))  # (K=4, N, inner)
        del self.dt_proj

        self.to_C = B_Gen3(K=1, dinner=self.d_inner)
        self.Cs_proj = (
            nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
        )
        self.Cs_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.Cs_proj], dim=0))  # (K=4, N, inner)
        del self.Cs_proj

        self.Bs_proj = (
            nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
        )
        self.Bs_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.Bs_proj], dim=0))  # (K=4, N, inner)
        del self.Bs_proj

        self.to_B = B_Gen3(K=1, dinner=self.d_inner)

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )  # 偏置尺寸等于原始影像
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        #
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):  # , rgb: torch.Tensor
        B, C, H, W = x.shape  # 输入影像尺寸
        L = H * W  # 将空间当作token，将不同空间排列视为token排序，光谱值作为编码值，但是光谱信息在第二维（transformer中token是才是第二维）
        K = 1  # 4

        x_hwwh = x.view(B, -1, L).view(B, 1, -1, L)  # 两个不同排列，左右上下，上下左右
        xs = x_hwwh  # (b, k, d, l) # 形成四个排列，左右上下，上下左右，右左下上，下上右左
        Cs = self.to_C(rearrange(xs, 'b k c (h w)->b (k c) h w', h=H, w=W))
        # Cs = xs
        Cs = torch.einsum("b k d l, k c d -> b k c l", Cs.view(B, K, -1, L),
                          self.Cs_proj_weight)  # 四个序列，利用线性变换进行四个不同的映射，注意此时光谱（第三个维度）变成dt+2*state

        dts = self.to_dt(rearrange(xs, 'b k c (h w)->b (k c) h w', h=H, w=W))
        # dts = xs
        dts = torch.einsum("b k d l, k c d -> b k c l", dts.view(B, K, -1, L),
                           self.dt_proj_weight)  # 四个序列，利用线性变换进行四个不同的映射，注意此时光谱（第三个维度）变成dt+2*state

        # x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L),
        #                      self.x_proj_weight)  # 四个序列，利用线性变换进行四个不同的映射，注意此时光谱（第三个维度）变成dt+2*state

        # dts, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state], dim=2)  # 映射的特征延光谱维进行分割生成dt，B，C

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)  # dt视为偏置

        # Bs = self.to_B(rearrange(xs, 'b k c (h w)->b (k c) h w', h=H, w=W))
        Bs = xs
        Bs = torch.einsum("b k d l, k c d -> b k c l", Bs.view(B, K, -1, L),
                          self.Bs_proj_weight)  # 四个序列，利用线性变换进行四个不同的映射，注意此时光谱（第三个维度）变成dt+2*state
        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # ht=A*ht+B*xs:b k*d k d_state
        # y = C*ht+D*xt: b k*d l
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  # 恢复复形状
        assert out_y.dtype == torch.float

        return out_y[:, 0]  # 左右上下，右左下上，上下左右，下上右左

    def forward(self, x: torch.Tensor, **kwargs):  # , rgb: torch.Tensor
        B, H, W, C = x.shape  # 输入影像尺寸

        xz = self.in_proj(x)  # 首先利用线性变换将原始维度转换为mamba内部维度
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d) 将x分割成两块一块留下备用
        del xz
        torch.cuda.empty_cache()
        x = x.permute(0, 3, 1, 2).contiguous()

        y = self.forward_core(x)  # 进行扫描, rgb
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # 将h*w展开
        y = self.out_norm(y)  # 层归一化
        y = y * F.silu(z)  # 留下的另一块形成权重进行加权

        out = self.out_proj(y)  # 恢复成原来维度
        if self.dropout is not None:
            out = self.dropout(out)
        return out
