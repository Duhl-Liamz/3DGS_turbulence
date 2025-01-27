import matplotlib
matplotlib.use('TkAgg')  # 更改为 TkAgg 后端
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# 假设输入数据
flow = np.load(f'E://Project_AI//3DGS//gsplat//examples//results_tilt_complex_4_test3//Raft_flow.npy')
flow = np.transpose(flow[0], (1, 2, 0))  # 转换为 (512, 512, 2)
# 使用最近邻插值进行尺寸扩展
scale_factor = 2  # 插值比例
output_tensor = zoom(flow, (scale_factor, scale_factor, 1), order=0)  # 最近邻插值 (order=0)

# 保存插值结果为 npy 文件
output_file = "upsampled_tensor.npy"
np.save(output_file, output_tensor)
print(f"Interpolated tensor saved to: {output_file}")

# 检查输出张量的形状
print("Original shape:", flow.shape)
print("Upsampled shape:", output_tensor.shape)

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 原始分辨率（第一个通道）
axes[0].imshow(flow[:, :, 0], cmap='gray', extent=(0, 1, 0, 1))
axes[0].set_title("Original Resolution (512x512)")

# 插值后分辨率（第一个通道）
axes[1].imshow(output_tensor[:, :, 0], cmap='gray', extent=(0, 1, 0, 1))
axes[1].set_title("Upsampled Resolution (1024x1024)")

plt.tight_layout()
plt.show()
