import matplotlib
matplotlib.use('TkAgg')  # 更改为 TkAgg 后端
import math
import os
import time
import nerfview
import viser
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
import torch

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
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image_1 = gt_image_1.to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image_1.shape[0], gt_image_1.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 512 # 散布范围与z值之间有一个关系

        x_init = (torch.rand(self.num_points, device=self.device, requires_grad=True)-0.5).unsqueeze(1) # rand是均匀分布，randn是高斯分布
        y_init = (torch.rand(self.num_points, device=self.device, requires_grad=True)-0.5).unsqueeze(1)
        # 固定 z 为 0
        z = torch.ones(self.num_points, device=self.device, requires_grad=False).unsqueeze(1)
        z = 250 * z  # 图片的视野范围大小由z值的大小确定，z值越大，范围越小，z值越小，图像越模糊
        x_init = bd * x_init
        y_init = bd * y_init
        # 通过缩放因子 bd 构建 means 变量
        self.means = torch.cat((x_init, y_init, z), dim=1)
        self.means = self.means.detach().requires_grad_()
        print(self.means.is_leaf)
        print('shape of means is ', self.means.shape)
        max_value = self.means.max()  # 获取最大值
        print(self.means)
        xx_init = torch.randn(self.num_points, device=self.device, requires_grad=True).unsqueeze(1)
        yy_init = torch.randn(self.num_points, device=self.device, requires_grad=True).unsqueeze(1)
        # 固定 z 为 0
        zz = torch.zeros(self.num_points, device=self.device, requires_grad=False).unsqueeze(1)

        # 通过缩放因子 bd 构建 means 变量
        self.scales = torch.cat((xx_init, yy_init, zz), dim=1).detach().requires_grad_()
        print(self.scales.is_leaf)
        print('shape of scales is ', self.scales.shape)
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)
        # self.quats = torch.ones((self.num_points, 4), device=self.device)  # 每个四元数的值都是 [1, 0, 0, 0]
        # w_init = torch.randn(self.num_points, device=self.device, requires_grad=True).unsqueeze(1)
        # z_init = torch.randn(self.num_points, device=self.device, requires_grad=True).unsqueeze(1)

        # 构造四元数 (x, y 固定为 0, w 和 z 作为可训练变量)
        # self.quats = torch.cat(
        #     (w_init, torch.zeros(self.num_points, device=self.device, requires_grad=False).unsqueeze(1), torch.zeros(self.num_points, device=self.device, requires_grad=False).unsqueeze(1), z_init),
        #     dim=1).detach().requires_grad_()

        theta = torch.randn(self.num_points, device=self.device, requires_grad=True).unsqueeze(1)
        theta = 180 * theta
        theta_rad = theta * (torch.pi / 180)
        w = torch.cos(theta_rad / 2)  # 标量部分
        z = torch.sin(theta_rad / 2)  # z 轴虚部
        self.quats = torch.cat((w, torch.zeros(self.num_points, device=self.device, requires_grad=False).unsqueeze(1), torch.zeros(self.num_points, device=self.device, requires_grad=False).unsqueeze(1), z), dim=1).detach().requires_grad_()
        print(self.quats.is_leaf)
        print('shape of quats is ', self.quats.shape)
        self.opacities = torch.ones((self.num_points), device=self.device)

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

        # self.means.requires_grad = True
        # self.scales.requires_grad = True
        # self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False


        # self.server = viser.ViserServer(port=cfg.port, verbose=False)
        # self.viewer = nerfview.Viewer(
        #     server=self.server,
        #     render_fn=self._viewer_render_fn,
        #     mode="training",
        # )


    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        model_type: Literal["3dgs", "2dgs"] = "2dgs",
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )

        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 2  # rasterization, backward
        im_end = []
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        if model_type == "3dgs":
            rasterize_fnc = rasterization
        elif model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs
        loss_values = []  # 用于保存每一步的loss值
        min_loss = float('inf')
        for iter in range(iterations):
            start = time.time()

            renders = rasterize_fnc(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                packed=False,
            )[0]

            out_img = renders[0]
            out_img = out_img
            torch.cuda.synchronize()
            times[0] += time.time() - start

            # MSE 和 SSIM 损失
            loss = 0.2 * compute_ssim(out_img, self.gt_image_1, self.device) + 0.8 * mse_loss(out_img,
                                                                                                self.gt_image_1)
            loss_values.append(loss.item())
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            optimizer.step()
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_output_img = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                best_output_img = Image.fromarray(best_output_img)
                out_dir = os.path.join(os.getcwd(), "results_tilt_complex_4_test3")
                os.makedirs(out_dir, exist_ok=True)
                best_output_img.save(f'{out_dir}/output_v3_{iter+1}.png')  # 保存为 PNG 文件
                print('best_output_img is saved')
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

        step_meta = {
            'means': self.means,
            'quats': self.quats,
            'scales': self.scales,
            'opacities': self.opacities,
            'rgbs': self.rgbs
        }
        out_dir = os.path.join(os.getcwd(), "results_tilt_complex_4_test3")
        os.makedirs(out_dir, exist_ok=True)
        torch.save(step_meta, f'{out_dir}/step_meta.pt')
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )

    # @torch.no_grad()
    # def _viewer_render_fn(
    #         self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    # ):
    #     """Callable function for the viewer."""
    #     W, H = img_wh
    #     c2w = camera_state.c2w
    #     K = camera_state.get_K(img_wh)
    #     c2w = torch.from_numpy(c2w).float().to(self.device)
    #     K = torch.from_numpy(K).float().to(self.device)
    #
    #     render_colors, _, _ = self.rasterize_splats(
    #         camtoworlds=c2w[None],
    #         Ks=K[None],
    #         width=W,
    #         height=H,
    #         sh_degree=self.cfg.sh_degree,  # active all SH degrees
    #         radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
    #     )  # [1, H, W, 3]
    #     return render_colors[0].cpu().numpy()


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 512,
    width: int = 512,
    num_points: int = 1048576, #262144
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 5000,
    lr: float = 0.01,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择设备
    vgg = models.vgg16(pretrained=True).features.eval().to(device)  # 将 VGG 模型移动到 GPU 或 CPU
    for param in vgg.parameters():
        param.requires_grad = False
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        # gt_image = torch.ones((height, width, 3)) * 1.0
        # # make top left and bottom right red, blue
        # gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        # gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])
        # gt_image = generate_image_with_circles(height, width)
        image_path_1 = './data_input_roundpoint/vertical_lines.png'  # 替换为您的PNG文件路径
        gt_image_1 = image_path_to_tensor(image_path_1)
        print('shape of gt_image_1:', gt_image_1.shape)
        print(f"Load image1: {gt_image_1.shape}")
    trainer = SimpleTrainer(gt_image_1=gt_image_1, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
        model_type=model_type,
    )


if __name__ == "__main__":
    tyro.cli(main)
