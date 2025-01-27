import matplotlib
matplotlib.use('TkAgg')  # 更改为 TkAgg 后端
import math
import os
from pathlib import Path
from typing import Literal, Optional
from torchmetrics import StructuralSimilarityIndexMeasure
import numpy as np
import torch
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

def gradient_loss(image1):
    """
    计算图像的梯度损失，用于提升图像的清晰度。

    Args:
        image1 (torch.Tensor): 输入图像，形状为 [B, C, H, W]。

    Returns:
        torch.Tensor: 梯度损失值。
    """
    # 确保输入图像是浮点类型
    if image1.dtype != torch.float32:
        image1 = image1.float()
    # 如果输入图像是3维的（H, W, C），增加批次维度
    if image1.ndimension() == 3:
        image1 = image1.unsqueeze(0).permute(0, 3, 1, 2)  # 增加批次维度，变为 [1, C, H, W]

    # 计算水平和垂直方向的梯度
    dx = image1[:, :, 1:, :] - image1[:, :, :-1, :]
    dy = image1[:, :, :, 1:] - image1[:, :, :, :-1]

    # 计算梯度的L1范数，表示图像的清晰度
    grad_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

    return grad_loss


# 读取PNG文件并转换为Tensor
def read_png_to_tensor(image_path):
    # 打开图片
    img = Image.open(image_path)

    # 转换为Tensor
    transform = transforms.ToTensor()
    img_tensor = transform(img)

    return img_tensor

def compute_ssim(preds, target, device):
    """
    计算 SSIM（结构相似性指数）值。
    如果输入图像没有批量维度（即形状为 [H, W, C]），则会自动添加批量维度。

    Args:
    - preds (torch.Tensor): 预测图像，形状为 (H, W, C)
    - target (torch.Tensor): 真实图像，形状为 (H, W, C)

    Returns:
    - torch.Tensor: 计算得到的 SSIM 值


    """
    # 检查 preds 和 target 的形状
    if preds.ndimension() == 3:  # 形状为 [H, W, C]
        preds = preds.unsqueeze(0)  # 增加批量维度，变为 [1, C, H, W]
    if target.ndimension() == 3:  # 形状为 [H, W, C]
        target = target.unsqueeze(0)  # 增加批量维度，变为 [1, C, H, W]

    preds = preds.permute(0, 3, 1, 2)  # 调整为 [B, C, H, W]
    target = target.permute(0, 3, 1, 2)  # 调整为 [B, C, H, W]

    # 使用 torchmetrics 计算 SSIM
    ssim_metric = StructuralSimilarityIndexMeasure()
    ssim_metric = ssim_metric.to(device)
    ssim_value = ssim_metric(preds, target)

    return 1 - ssim_value

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image_1: Tensor,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image_1 = gt_image_1.to(device=self.device)
        fov_x = math.pi / 2.0
        self.H, self.W = gt_image_1.shape[0], gt_image_1.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        d = 3
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)
        self.viewmat.requires_grad = False

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        model_type: Literal["3dgs", "2dgs"] = "3dgs",
    ):
        mse_loss = torch.nn.MSELoss()
        splat = torch.load(f'E://Project_AI//3DGS//gsplat//examples//results_tilt_complex_4_test3//step_meta_grid_1024_3.pt')
        # flow = np.load(f'E://Project_AI//3DGS//gsplat//examples//results_tilt_complex_4_test3//Raft_flow_1024.npy')
        # print('shape of flow:', flow.shape)

        # reshaped_data = np.squeeze(flow)  # 移除维度为 1 的轴，结果形状为 (2, 512, 512)
        # reshaped_data = np.transpose(reshaped_data, (1, 2, 0))  # 转换为 (512, 512, 2)

        # 生成四分之一图像横向偏移10像素的光流
        reshaped_data = np.zeros((1024, 1024, 2), dtype=np.float32)
        reshaped_data[:512, 512:, 0] = -10  # channel1
        reshaped_data[:512, 512:, 1] = 0  # channel2

        reshaped_data_flattened = reshaped_data.reshape(-1, 2)
        flow = torch.from_numpy(reshaped_data_flattened)
        flow = flow.to(self.device)



        means = splat['means']
        print(means)
        quats = splat['quats']
        print('shape of quats: ', quats.shape)
        scales = splat['scales']
        opacities = splat['opacities']
        rgbs = splat['rgbs']

        means = means.detach()  # 不再计算梯度
        # means[..., :2] += -flow_train

        # 高斯球位置与光流配准
        x_indices = torch.arange(1024, device='cuda')
        y_indices = torch.arange(1024, device='cuda')
        grid_x, grid_y = torch.meshgrid(x_indices, y_indices, indexing="ij")

        # 映射到 [-255, 255]
        x_coords = (grid_x / 1023) * 510 - 255
        y_coords = (grid_y / 1023) * 510 - 255

        batch_size = 256

        # 合并坐标，形状为 (4096, 4096, 2)
        mapped_coords = torch.stack((x_coords, y_coords), dim=-1)

        # 2. 将张量1的 xy 与张量2的坐标进行最近邻匹配
        # 展平张量2的坐标
        mapped_coords_flat = mapped_coords.view(-1, 2)  # 形状 (4096*4096, 2)

        # 计算欧几里得距离
        tensor1_xy = means[:, :2]  # 取 tensor1 的 xy 部分
        num_points = tensor1_xy.shape[0]
        nearest_indices = []


        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            batch_tensor1 = tensor1_xy[start:end]
            distances = torch.cdist(batch_tensor1, mapped_coords_flat)

            # 找到最近邻的索引
            nearest_indices.append(torch.argmin(distances, dim=-1))

        nearest_indices = torch.cat(nearest_indices, dim=0)  # 合并所有批次结果

        # 3. 根据最近邻索引获取对应的 dx, dy
        dx_dy_values = flow.view(-1, 2)[nearest_indices]  # 最近邻对应的 dx, dy 值

        # 4. 将 dx, dy 与 tensor1 的 xy 相加
        updated_xy = tensor1_xy - dx_dy_values

        # 如果需要更新 tensor1：
        tensor1_updated = means.clone()
        tensor1_updated[:, :2] = updated_xy
        means[:, :2] = tensor1_updated[:, :2]

        means_train = torch.nn.Parameter(means)
        quats_train = torch.nn.Parameter(quats)
        scales_train = torch.nn.Parameter(scales)
        opacities_train = torch.nn.Parameter(opacities)
        rgbs_train = torch.nn.Parameter(rgbs)
        # flow_train = torch.rand(16777216, 2, device=self.device)
        # print('shape of flow_train: ', flow_train.shape)
        optimizer = optim.Adam(
            [means_train, quats_train, scales_train, opacities_train, rgbs_train], lr
        )
        print('data is loaded')
        print('shape of splat', splat['means'].shape)
        # print('shape of flow', flow.shape)

        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )
        loss_values = []  # 用于保存每一步的loss值



        if model_type == "3dgs":
            rasterize_fnc = rasterization
        elif model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs
        min_loss = float('inf')
        # min_loss = 0
        for iter in range(iterations):
            renders = rasterize_fnc(
                means,
                quats / quats.norm(dim=-1, keepdim=True),
                scales,
                torch.sigmoid(opacities),
                torch.sigmoid(rgbs),
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                packed=False,
            )[0]
            out_img = renders[0]
            print('shape of out_img', out_img.shape)
            # reconstructed_image1 = warp_back(out_img, flow)
            loss = 0.2 * compute_ssim(out_img, self.gt_image_1, self.device) + 0.8 * mse_loss(out_img,
                                                                                          self.gt_image_1)
            loss_values.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize()
            optimizer.step()
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_output_img = (out_img.detach().cpu().numpy() * 255).astype(np.uint8) # 此处保存的原始结果，非tilt以后的结果，其应该与原始分布的投影结果保持一致，即与0.png保持一致
                best_output_img = Image.fromarray(best_output_img)
                out_dir = os.path.join(os.getcwd(), "results_tilt_complex_4_projection_test12")
                os.makedirs(out_dir, exist_ok=True)
                best_output_img.save(f'{out_dir}/origin_{iter + 1}.png')  # 保存为 PNG 文件
                print('best_output_img is saved')
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
        step_meta = {
            'means': means,
            'quats': quats,
            'scales': scales,
            'opacities': opacities,
            'rgbs': rgbs
        }
        out_dir = os.path.join(os.getcwd(), "results_tilt_complex_4_projection_test12")
        os.makedirs(out_dir, exist_ok=True)
        torch.save(step_meta, f'{out_dir}/step_meta.pt')
        print('pt is saved')

def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 512,
    width: int = 512,
    num_points: int = 200000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1,
    lr: float = 1e-4,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择设备
    vgg = models.vgg16(pretrained=True).features.eval().to(device)  # 将 VGG 模型移动到 GPU 或 CPU
    for param in vgg.parameters():
        param.requires_grad = False
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        image_path_1 = './data_input_roundpoint/vertical_lines.png'  # 替换为您的PNG文件路径
        gt_image_1 = image_path_to_tensor(image_path_1)
        print(f"Load image1: {gt_image_1.shape}")

    trainer = SimpleTrainer(gt_image_1=gt_image_1)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
        model_type=model_type,
    )


if __name__ == "__main__":
    tyro.cli(main)
