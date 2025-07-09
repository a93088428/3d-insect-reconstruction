import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from PIL import Image
from transformers import Dinov2Model
from scipy.spatial import cKDTree
from torchvision import transforms

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class Dataset:
    def __init__(self, conf, vit_batch_size=4):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.vit_batch_size = vit_batch_size

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))

        valid_indices = []
        for idx in range(len(self.images_lis)):
            if f'world_mat_{idx}' in camera_dict and f'scale_mat_{idx}' in camera_dict:
                valid_indices.append(idx)
            else:
                print(f"Warning: Missing camera or scale data for index {idx}. Skipping.")

        self.images_lis = [self.images_lis[idx] for idx in valid_indices]
        self.masks_lis = [self.masks_lis[idx] for idx in valid_indices]
        self.n_images = len(self.images_lis)

        # Load images in RGB and normalize to [0, 1]
        self.images_np = np.stack([cv.cvtColor(cv.imread(im_name), cv.COLOR_BGR2RGB) for im_name in self.images_lis]) / 255.0
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # (n_images, H, W, 3)
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()

        self.world_mats_np = [camera_dict[f'world_mat_{idx}'].astype(np.float32) for idx in valid_indices]
        self.scale_mats_np = [camera_dict[f'scale_mat_{idx}'].astype(np.float32) for idx in valid_indices]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        # Initialize DINOv2 for feature extraction
        self.vit_model = Dinov2Model.from_pretrained('facebook/dinov2-base')
        self.patch_size = 14
        self.feature_dim = 768

        # Preprocessing for DINOv2
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Compute feature maps and padded sizes
        self.feature_maps = []
        self.padded_sizes = []
        self.compute_feature_maps()

        # Compute neighboring views
        self.compute_neighboring_views(k=5)

        print('Load data: End')

    def pad_to_patch_multiple(self, img):
        B, C, H, W = img.shape
        pad_h = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size - H
        pad_w = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size - W
        padding_h_top = pad_h // 2
        padding_h_bottom = pad_h - padding_h_top
        padding_w_left = pad_w // 2
        padding_w_right = pad_w - padding_w_left
        img_padded = F.pad(img, (padding_w_left, padding_w_right, padding_h_top, padding_h_bottom), mode='constant', value=0)
        return img_padded, (img_padded.shape[2], img_padded.shape[3])  # (H_padded, W_padded)

    def compute_feature_maps(self):
        self.vit_model.eval()
        self.vit_model.to(self.device)
        self.feature_maps = []
        self.padded_sizes = []
        with torch.no_grad():
            for img in self.images:
                # Permute from (H, W, 3) to (3, H, W)
                img = img.permute(2, 0, 1)  # (3, H, W)
                # Apply normalization
                img = self.preprocess(img)
                # Add batch dimension
                img = img.unsqueeze(0)  # (1, 3, H, W)
                img_padded, padded_size = self.pad_to_patch_multiple(img)
                # Move to device
                img_padded = img_padded.to(self.device)
                features = self.vit_model(img_padded).last_hidden_state[:, 1:]  # Exclude CLS token
                h_padded, w_padded = padded_size
                h_feat, w_feat = h_padded // self.patch_size, w_padded // self.patch_size
                feature_map = features.view(1, h_feat, w_feat, self.feature_dim).permute(0, 3, 1, 2).squeeze(0)
                self.feature_maps.append(feature_map.cpu())  # Store on CPU to save GPU memory
                self.padded_sizes.append(padded_size)

    def compute_neighboring_views(self, k=5):
        centers = self.pose_all[:, :3, 3].cpu().numpy()
        tree = cKDTree(centers)
        _, indices = tree.query(centers, k=k + 1)
        self.neighbor_indices = indices[:, 1:]  # Exclude self

    def gen_rays_at(self, img_idx, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        if img_idx >= self.n_images:
            raise ValueError(f"Invalid image index {img_idx}. Total images: {self.n_images}")

        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size], device='cpu')
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size], device='cpu')
        color = self.images[img_idx][(pixels_y, pixels_x)]
        mask = self.masks[img_idx][(pixels_y, pixels_x)]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)
        return torch.cat([rays_o, rays_v, color.to(self.device), mask[:, :1].to(self.device)], dim=-1)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
        rays_o = trans[None, None, :3].expand(rays_v.shape)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def save_feature_maps(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for i, feat_map in enumerate(self.feature_maps):
            torch.save(feat_map.cpu(), os.path.join(save_dir, f'feature_map_{i}.pt'))