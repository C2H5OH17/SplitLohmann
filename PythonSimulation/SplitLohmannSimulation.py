import os
import numpy as np
import skimage.io
import cv2
import matplotlib.pyplot as plt
from params import Params
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# 导入compute.py中的函数
from compute_static_python.compute import digitize, load_images, crop_grayscale_image, crop_color_image, fit_images, \
    compute_phase_mask, save2disk

# 设置中文字体
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "KaiTi"]

# 定义输入输出路径
input_folder = os.path.join('scenes', 'png')
output_folder = os.path.join('results', 'SYCCastleCitySimulation')
others_folder = os.path.join(output_folder, 'Others')
texture_map_name = 'CastleCity_TextureMap.png'
diopter_map_name = 'CastleCity_DiopterMap.png'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(others_folder):
    os.makedirs(others_folder)

# 定义参数
discretize = True
numdepths = 50
params = Params()
params.W = 4.0  # 工作范围（4屈光度）
oled_shape = (1200, 1920)  # OLED分辨率
slm_shape = (1080, 1920)  # SLM分辨率
print('OLED目标图像形状:', oled_shape)
print('SLM目标图像形状:', slm_shape)

# 使用绝对路径加载HomographyMatrixOLED2SLM.npy文件
homography_matrix_path = os.path.join('compute_static_python', 'data', 'HomographyMatrixOLED2SLM.npy')
try:
    H = np.load(homography_matrix_path)
    print("成功加载Homography矩阵")
except FileNotFoundError:
    # 如果文件不存在，创建一个恒等矩阵作为示例
    H = np.eye(3)
    print("未找到Homography矩阵，使用恒等矩阵进行演示")

# 加载图像
print("正在加载纹理图和深度图...")
texture_map, diopter_map = load_images(input_folder, discretize, numdepths, texture_map_name, diopter_map_name)

# 裁剪和变形图像
print("正在裁剪和变形图像...")
texture_map_out, diopter_map_out = fit_images(H, texture_map, diopter_map, oled_shape, slm_shape)

# 计算相位掩码
print("正在计算相位掩码...")
phase_mask = compute_phase_mask(diopter_map_out, params, diopter_map_name)

# 保存中间结果到Others文件夹
save2disk(others_folder, 'OLED', texture_map_out, texture_map_name)
save2disk(others_folder, 'SLM', diopter_map_out, diopter_map_name)
save2disk(others_folder, 'SLM', phase_mask, 'phase_mask_' + diopter_map_name)


# 角谱衍射理论模型实现
def angular_spectrum_propagation(field, dx, dy, wavelength, distance):
    """
    使用角谱法模拟光波传播（论文中的角谱衍射理论）
    参数:
    field: 输入光场分布
    dx, dy: 空间采样间隔(m)
    wavelength: 波长(m)
    distance: 传播距离(m)
    返回:
    propagated_field: 传播后的光场分布
    """
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength  # 波数
    phase_factor = np.exp(1j * k * distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))
    phase_factor[np.isnan(phase_factor)] = 0  # 处理数值计算中的奇点
    # k = 2 * np.pi / wavelength
    # spatial_freq = np.sqrt((wavelength * FX) ** 2 + (wavelength * FY) ** 2)
    # valid = spatial_freq <= 1.0
    # phase_factor = np.exp(1j * k * distance * np.sqrt(1 - spatial_freq[valid] ** 2))    # 处理数值计算中的奇点
    # phase_factor = np.full_like(FX, np.nan)
    # phase_factor[valid] = phase_factor

    field_freq = fft2(fftshift(field))
    field_freq_propagated = field_freq * phase_factor
    propagated_field = ifftshift(ifft2(field_freq_propagated))
    return propagated_field

#FIXME
# def angular_spectrum_propagation(...):
#     # 增强数值稳定性
#     k = 2 * np.pi / wavelength
#     spatial_freq = np.sqrt((wavelength * FX) ** 2 + (wavelength * FY) ** 2)
#     valid = spatial_freq <= 1.0
#     phase_factor = np.exp(1j * k * distance * np.sqrt(1 - spatial_freq[valid] ** 2))
#     phase_factor = np.full_like(FX, np.nan)
#     phase_factor[valid] = phase_factor

# 菲涅尔衍射理论模型实现
def fresnel_propagation(field, dx, dy, wavelength, distance):
    """
    使用菲涅耳传播方法模拟光波传播（论文中从P5到眼睛的传播）
    参数:
    field: 输入光场分布
    dx, dy: 空间采样间隔(m)
    wavelength: 波长(m)
    distance: 传播距离(m)
    返回:
    propagated_field: 传播后的光场分布
    """
    ny, nx = field.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    k = 2 * np.pi / wavelength
    H = np.exp(1j * k * distance) * np.exp(1j * k / (2 * distance) * (X ** 2 + Y ** 2))
    U1 = fft2(fftshift(field))
    U2 = U1 * H
    propagated_field = ifftshift(ifft2(U2))
    return propagated_field


def simulate_split_lohmann_system(texture_map, phase_mask, params, propagation_distance, num_trials=10):
    """
    模拟Split-Lohmann系统中的光波传播，包括多色光处理和空间非相干性模拟
    参数:
    texture_map: OLED纹理图
    phase_mask: SLM相位掩码
    params: 系统参数
    propagation_distance: 从SLM到观察平面的距离(m)
    num_trials: 空间非相干性模拟的试验次数
    返回:
    intensity: 观察平面的光强分布
    """
    intensities = []

    # 确保纹理图和相位掩码尺寸一致
    if texture_map.shape[:2] != phase_mask.shape:

        # 调整纹理图尺寸以匹配相位掩码
        texture_map_resized = np.zeros((phase_mask.shape[0], phase_mask.shape[1], texture_map.shape[2]))
        for channel in range(texture_map.shape[2]):
            texture_map_resized[..., channel] = cv2.resize(
                texture_map[..., channel],
                (phase_mask.shape[1], phase_mask.shape[0]),  # 注意尺寸顺序: (宽度, 高度)
                interpolation=cv2.INTER_LINEAR
            )
        texture_map = texture_map_resized

    # 不同颜色通道的波长
    wavelengths = [630e-9, 530e-9, 470e-9]  # RGB对应波长

    for trial in range(num_trials):
        # 空间非相干性模拟：添加随机相位
        random_phase = np.random.uniform(0, 2 * np.pi, phase_mask.shape)
        phase_mask_with_random = phase_mask + random_phase / (2 * np.pi)

        intensity_channel = []
        for channel in range(texture_map.shape[2]):
            monochrome_texture = texture_map[..., channel]
            dx = params.oledPitch  # OLED像素间距作为采样间隔
            dy = params.oledPitch
            wavelength = wavelengths[channel]  # 设置对应通道的波长

            # 组合纹理和相位掩码（SLM对光场进行相位调制）
            field_at_slm = monochrome_texture * np.exp(1j * 2 * np.pi * phase_mask_with_random)

            # 使用角谱法模拟从SLM到观察平面的传播
            propagated_field = angular_spectrum_propagation(
                field_at_slm, dx, dy, wavelength, propagation_distance
            )

            # 从观察平面（P5）到眼睛的传播
            distance_to_eye = 25e-3  # 视网膜与眼内晶状体相距25毫米
            field_at_eye = fresnel_propagation(
                propagated_field, dx, dy, wavelength, distance_to_eye
            )

            # 计算光强分布
            intensity = np.abs(field_at_eye) ** 2
            intensity_channel.append(intensity)

        intensities.append(np.stack(intensity_channel, axis=-1))

    #FIXME
    # # 对各次试验的强度取平均，模拟非相干光
    # final_intensity = np.mean(intensities, axis=0)
    # final_intensity = final_intensity / np.max(final_intensity)  # 归一化强度
    # return final_intensity
    # 保留原始强度分布（不全局归一化）
    final_intensity = np.mean(intensities, axis=0)
    # 局部归一化增强对比度
    for channel in range(final_intensity.shape[2]):
        img = final_intensity[..., channel]
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        final_intensity[..., channel] = img

    # 对各次试验的强度取平均，模拟非相干光（论文中的空间非相干性处理）
    final_intensity = np.mean(intensities, axis=0)
    final_intensity = final_intensity / np.max(final_intensity)  # 归一化强度
    return final_intensity

#FIXME
# # 模拟光线传播并生成不同深度的图像

# print("正在模拟不同深度的光线传播...")
# eye_focus_depths = [0.1, 0.2, 0.3, 0.5]  # 4个不同深度（单位：米）
#
# # 人眼晶状体焦距（米）
# f_eye = 25e-3
#
# for idx, depth in enumerate(eye_focus_depths):
#     # 根据论文公式计算传播距离（有效焦距）
#     # 屈光度 = 1/深度（米）
#     diopter = 1 / depth if depth > 0 else 0
#     # 有效焦距 = 1 / (屈光度 + 1/f_eye)
#     propagation_distance = 1 / (diopter + 1 / f_eye)
#
#     print(f"深度 {depth} 米 对应屈光度 {diopter:.2f} D，传播距离 {propagation_distance * 1000:.2f} mm")
#
#     # 执行仿真
#     intensity = simulate_split_lohmann_system(
#         texture_map_out, phase_mask, params, propagation_distance
#     )
#
#     # 保存结果...

# 模拟光线传播并生成不同深度的图像
print("正在模拟不同深度的光线传播...")
eye_focus_depths = [0.1, 0.2, 0.3, 0.5]  # 4个不同深度（单位：米），对应不同焦距

# for idx, depth in enumerate(eye_focus_depths):
#     # 根据深度计算对应的传播距离（论文中的焦距与深度关系）
#     propagation_distance = depth  # 简化处理，实际应根据论文公式计算

for idx, depth in enumerate(eye_focus_depths):
    # 根据论文公式计算传播距离（有效焦距）
    # 屈光度 = 1/深度（米）
    diopter = 1 / depth if depth > 0 else 0
    # 有效焦距 = 1 / (屈光度 + 1/f_eye)
    propagation_distance = 1 / (diopter + 1 / params.f_eye)

    # 执行仿真
    intensity = simulate_split_lohmann_system(
        texture_map_out, phase_mask, params, propagation_distance
    )

    # 保存不同深度的图像到输出文件夹
    save_path = os.path.join(output_folder, f'eye_view_depth_{idx + 1}.png')
    intensity_uint8 = (np.round(intensity * 255)).astype(np.uint8)
    skimage.io.imsave(save_path, intensity_uint8)
    print(f'已保存深度为{depth}米的图像到: {save_path}')

    # 保存中间过程的光强分布图到Others文件夹
    plt.figure(figsize=(10, 8))
    plt.imshow(intensity, cmap='viridis')
    plt.title(f'深度{depth}米处的光强分布')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(others_folder, f'intensity_distribution_depth_{idx + 1}.png'))
    plt.close()

# 可视化不同深度的图像（汇总到一个图中）
print("正在生成汇总可视化结果...")
plt.figure(figsize=(16, 12))
for i, depth in enumerate(eye_focus_depths):
    img_path = os.path.join(output_folder, f'eye_view_depth_{i + 1}.png')
    img = skimage.io.imread(img_path)
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(f'深度 {depth} 米')
    plt.axis('off')
plt.tight_layout()
summary_path = os.path.join(output_folder, 'summary_eye_views.png')
plt.savefig(summary_path)
plt.close()
print(f'已保存汇总可视化结果到: {summary_path}')

print("仿真模拟完成！所有结果已保存到指定文件夹。")