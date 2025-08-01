import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=[1.0, 1.0, 2.0],
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 feature_dim=256,
                 device=None):
        super(SDFNetwork, self).__init__()

        if max(skip_in, default=-1) >= n_layers or min(skip_in, default=n_layers) < 0:
            raise ValueError(f"skip_in indices {skip_in} must be within [0, {n_layers-1}]")
        if not skip_in or max(skip_in) >= n_layers - 1:
            raise ValueError(f"skip_in {skip_in} must contain valid indices less than n_layers-1 ({n_layers-1})")

        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.skip_in = skip_in
        self.multires = multires
        self.feature_dim = feature_dim

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            self.embedded_input_dim = input_ch
        else:
            self.embed_fn_fine = lambda x: x
            self.embedded_input_dim = d_in

        dims = [self.embedded_input_dim + feature_dim] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        self.scale = torch.tensor(scale, dtype=torch.float32).to(device) if device else torch.tensor(scale, dtype=torch.float32)

        self.layers = nn.ModuleList()
        for l in range(self.num_layers - 1):
            in_dim = dims[l]
            if l in self.skip_in:
                in_dim += self.embedded_input_dim
            out_dim = dims[l + 1]
            lin = nn.Linear(in_dim, out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(self.embedded_input_dim - 3):], 0.0) if self.embedded_input_dim > 3 else None
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            self.layers.append(lin)

        self.feature_input_dim = 768  # DINOv2 feature dim
        self.linear_pe = nn.Linear(self.embedded_input_dim, self.feature_input_dim)
        self.linear_k = nn.Linear(self.feature_input_dim, self.feature_input_dim)
        self.linear_fused = nn.Linear(self.feature_input_dim, self.feature_dim)

        self.activation = nn.Softplus(beta=100)
        self.to(device)

    def forward(self, inputs, features=None):
        inputs = inputs * self.scale
        x = self.embed_fn_fine(inputs)
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        else:
            # Concatenate zeros for features when features is None
            zeros = torch.zeros(inputs.shape[0], self.feature_dim, device=inputs.device)
            x = torch.cat([x, zeros], dim=-1)

        for l, layer in enumerate(self.layers):
            if l in self.skip_in:
                x = torch.cat([x, self.embed_fn_fine(inputs)], 1) / np.sqrt(2)
            x = layer(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, features=None):
        return self.forward(x, features)[:, :1]

    def gradient(self, x, features=None):
        x.requires_grad_(True)
        y = self.sdf(x, features)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True,
                 image_feature_dim=256,
                 device=None):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature + image_feature_dim] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embed_fn_view, input_ch = get_embedder(multires_view)
            self.embedview_fn = embed_fn_view
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, points, normals, view_dirs, feature_vectors, features):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors, features], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors, features], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors, features], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x

class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False,
                 device=None):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.to(device)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, device=None):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val).to(device)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)