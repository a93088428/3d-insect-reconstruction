import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def sample_pdf(bins, weights, n_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples

class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        batch_size, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)
        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))
        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)
        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    feature_maps,
                    poses,
                    intrinsics,
                    padded_sizes,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape
        device = rays_o.device

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dirs = rays_d[:, None, :].expand(pts.shape)

        # Project points to neighboring views and sample features
        k = len(feature_maps)
        pts_hom = torch.cat([pts, torch.ones(batch_size, n_samples, 1, device=device)], dim=-1)
        pts_hom = pts_hom.unsqueeze(0).expand(k, -1, -1, -1)
        poses_inv = torch.inverse(poses)
        pts_cam = torch.einsum('kij,klmj->klmi', poses_inv, pts_hom)[..., :3]
        u = (intrinsics[:, 0, 0].view(k, 1, 1) * pts_cam[..., 0] / (pts_cam[..., 2] + 1e-6)) + intrinsics[:, 0, 2].view(k, 1, 1)
        v = (intrinsics[:, 1, 1].view(k, 1, 1) * pts_cam[..., 1] / (pts_cam[..., 2] + 1e-6)) + intrinsics[:, 1, 2].view(k, 1, 1)
        w_padded = torch.tensor([p[0] for p in padded_sizes], device=device).view(k, 1, 1)
        h_padded = torch.tensor([p[1] for p in padded_sizes], device=device).view(k, 1, 1)
        x_grid = 2 * (u / (w_padded - 1)) - 1
        y_grid = 2 * (v / (h_padded - 1)) - 1

        features_all = []
        for i in range(k):
            grid = torch.stack([x_grid[i], y_grid[i]], dim=-1).view(1, batch_size * n_samples, 1, 2)
            features = F.grid_sample(feature_maps[i].unsqueeze(0), grid, align_corners=True)
            features = features.view(batch_size, n_samples, -1)
            features_all.append(features)
        features_all = torch.stack(features_all, dim=2)  # (batch_size, n_samples, k, C)

        # Cross-attention with torch.bmm fix and debug prints
        C = features_all.shape[-1]  # e.g., 768
        pe = sdf_network.embed_fn_fine(pts.reshape(-1, 3)).reshape(batch_size, n_samples, -1)
        query = sdf_network.linear_pe(pe)  # (batch_size, n_samples, 768)
        keys = sdf_network.linear_k(features_all)  # (batch_size, n_samples, k, 768)
        values = features_all  # (batch_size, n_samples, k, 768)

        # Compute attention logits
        query_reshaped = query.view(batch_size * n_samples, 1, query.shape[-1])  # (b*n, 1, 768)
        keys_reshaped = keys.view(batch_size * n_samples, k, keys.shape[-1])  # (b*n, k, 768)
        attn_logits = torch.bmm(query_reshaped, keys_reshaped.transpose(1, 2))  # (b*n, 1, k)
        attn_logits = attn_logits.view(batch_size, n_samples, k) / (C ** 0.5)  # (b, n, k)
        attn = F.softmax(attn_logits, dim=-1)  # (b, n, k)

        # Debug prints for attention logits
        print("query_reshaped shape:", query_reshaped.shape)
        print("keys_reshaped shape:", keys_reshaped.shape)
        print("attn_logits shape:", attn_logits.shape)

        # Compute fused features
        attn_reshaped = attn.view(batch_size * n_samples, k, 1)  # (b*n, k, 1)
        values_reshaped = values.view(batch_size * n_samples, k, C)  # (b*n, k, 768)
        fused = torch.bmm(attn_reshaped.transpose(1, 2), values_reshaped)  # (b*n, 1, 768)
        fused = fused.view(batch_size, n_samples, C)  # (b, n, 768)
        fused_proj = sdf_network.linear_fused(fused)  # (b, n, 256)

        # Debug prints for fused features
        print("attn_reshaped shape:", attn_reshaped.shape)
        print("values_reshaped shape:", values_reshaped.shape)
        print("fused shape:", fused.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        sdf_nn_output = sdf_network(pts, features=fused_proj.reshape(-1, fused_proj.shape[-1]))
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        gradients = sdf_network.gradient(pts, features=fused_proj.reshape(-1, fused_proj.shape[-1])).squeeze()
        sampled_color = color_network(pts, gradients, dirs, feature_vector, fused_proj.reshape(-1, fused_proj.shape[-1])).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] + background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights_sum)
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, feature_maps=None, poses=None, intrinsics=None, padded_sizes=None, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = rays_o.shape[0]
        sample_dist = 2.0 / self.n_samples
        z_vals_base = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device)
        z_vals = z_vals_base.unsqueeze(0).repeat(batch_size, 1)
        z_vals = near + (far - near) * z_vals
        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside_base = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device=rays_o.device)
            z_vals_outside = z_vals_outside_base.unsqueeze(0).repeat(batch_size, 1)
        n_samples = self.n_samples
        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=rays_o.device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples
            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]], device=rays_o.device)
                z_vals_outside = lower + (upper - lower) * t_rand
        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples
        background_alpha = None
        background_sampled_color = None
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))
            n_samples = self.n_samples + self.n_importance
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)
            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    feature_maps=feature_maps,
                                    poses=poses,
                                    intrinsics=intrinsics,
                                    padded_sizes=padded_sizes,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)
        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)
        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))