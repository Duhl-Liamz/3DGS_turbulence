import matplotlib
matplotlib.use('TkAgg')  # 更改为 TkAgg 后端
import math
import os
from pathlib import Path
from typing import Literal, Optional
from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np
import torch
import torch.nn as nn
import tyro
from PIL import Image
from torch import Tensor, optim
from gsplat import rasterization, rasterization_2dgs
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F


def warp_back(image2, flow2):
    """
    Warp image2 back to image1 using the inverse of flow2.

    Parameters:
    - image2: numpy array of shape [H, W, C], RGB image.
    - flow2: numpy array of shape [H, W, 2], optical flow (u, v), mapping from image1 to image2.

    Returns:
    - reconstructed_image1: numpy array of shape [H, W, C], reconstructed image1.
    """
    # Step 1: Convert image2 and flow2 to PyTorch tensors
    # image2 = torch.from_numpy(image2.transpose((2, 0, 1))).float().unsqueeze(0)  # [1, C, H, W]
    image2 = image2.permute(2, 0, 1).float().unsqueeze(0)
    # flow2 = torch.from_numpy(flow2.transpose(2, 0, 1)).float().unsqueeze(0)  # [1, 2, H, W]

    B, C, H, W = image2.size()

    # Step 2: Create mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()  # [B, 2, H, W]

    # Step 3: Compute the inverse flow
    vgrid = grid - flow2  # Reverse flow direction

    # Step 4: Normalize grid to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]

    # Step 5: Warp the image using grid_sample
    image2 = image2.cuda()
    vgrid = vgrid.cuda()
    reconstructed_image1 = F.grid_sample(image2, vgrid, align_corners=True)

    # Step 6: Generate a valid mask
    mask = torch.ones_like(reconstructed_image1, device=reconstructed_image1.device)
    mask = F.grid_sample(mask, vgrid, align_corners=True)
    mask = (mask >= 0.9999).float()  # Mask invalid regions

    # Step 7: Apply the mask to the output
    reconstructed_image1 = reconstructed_image1 * mask
    reconstructed_image1 = reconstructed_image1.squeeze(0).permute(1, 2, 0)
    # reconstructed_image1 = reconstructed_image1.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    # reconstructed_image1 = np.clip(reconstructed_image1, 0, 255).astype(np.uint8)  # Clip to valid range

    return reconstructed_image1

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

flow = np.load(f'E://Project_AI//3DGS//gsplat//examples//results_tilt_complex_4_test3//Raft_flow.npy')
# flow = flow.transpose(2, 0, 1)[np.newaxis, ...]
# print('shape of flow:', flow.shape)
image_path_1 = './data_input_roundpoint/vertical_lines.png'  # 替换为您的PNG文件路径
gt_image_1 = image_path_to_tensor(image_path_1)

print(f"Load image1: {gt_image_1.shape}")
reconstructed_image1 = warp_back(gt_image_1, flow)
best_output_img = (reconstructed_image1.detach().cpu().numpy() * 255).astype(
    np.uint8)  # 此处保存的原始结果，非tilt以后的结果，其应该与原始分布的投影结果保持一致，即与0.png保持一致
best_output_img = Image.fromarray(best_output_img)
out_dir = os.path.join(os.getcwd(), "results_tilt_complex_4_projection_test12")
os.makedirs(out_dir, exist_ok=True)
best_output_img.save(f'{out_dir}/recon_gridsample.png')  # 保存为 PNG 文件
