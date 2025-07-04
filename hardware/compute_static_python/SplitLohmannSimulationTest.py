import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import cv2

# 角谱衍射理论模型实现
def angular_spectrum_propagation(field, dx, dy, wavelength, distance):
    """
    使用角谱法模拟光波传播

    参数:
    field: 输入光场分布
    dx, dy: 空间采样间隔(m)
    wavelength: 波长(m)
    distance: 传播距离(m)

    返回:
    propagated_field: 传播后的光场分布
    """
    # 输入光场的尺寸
    ny, nx = field.shape

    # 计算空间频率
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    # 计算角谱相位因子
    k = 2 * np.pi / wavelength  # 波数
    phase_factor = np.exp(1j * k * distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))

    # 处理数值计算中的奇点
    phase_factor[np.isnan(phase_factor)] = 0

    # 傅里叶变换到频域
    field_freq = fft2(fftshift(field))

    # 应用相位因子
    field_freq_propagated = field_freq * phase_factor

    # 逆傅里叶变换回空间域
    propagated_field = ifftshift(ifft2(field_freq_propagated))

    return propagated_field

# 菲涅耳传播方法
def fresnel_propagation(field, dx, dy, wavelength, distance):
    """
    使用菲涅耳传播方法模拟光波传播

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
    模拟Split-Lohmann系统中的光波传播

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

    for _ in range(num_trials):
        # 空间非相干性模拟：添加随机相位
        random_phase = np.random.uniform(0, 2 * np.pi, phase_mask.shape)
        phase_mask_with_random = phase_mask + random_phase / (2 * np.pi)

        # 完善多色光处理：对每个通道进行处理
        intensity_channel = []
        for channel in range(texture_map.shape[2]):
            monochrome_texture = texture_map[..., channel]

            # 调整monochrome_texture的形状与phase_mask一致
            monochrome_texture = crop_grayscale_image(monochrome_texture, phase_mask.shape)

            # 计算空间采样间隔
            dx = params.oledPitch  # OLED像素间距作为采样间隔
            dy = params.oledPitch

            # 组合纹理和相位掩码(假设SLM对光场进行相位调制)
            # 这里假设光从OLED出发，经过SLM相位调制
            field_at_slm = monochrome_texture * np.exp(1j * 2 * np.pi * phase_mask_with_random)

            # 使用角谱法模拟从SLM到观察平面的传播
            propagated_field = angular_spectrum_propagation(
                field_at_slm, dx, dy, params.lbda, propagation_distance
            )

            # 从观察平面（P5）到眼睛的传播
            distance_to_eye = 25e-3  # 假设视网膜与眼内晶状体相距25毫米
            # 考虑二次相位项（此处简化，可根据具体情况调整）
            k = 2 * np.pi / params.lbda
            x = np.arange(propagated_field.shape[1]) * dx
            y = np.arange(propagated_field.shape[0]) * dy
            X, Y = np.meshgrid(x, y)
            quadratic_phase = np.exp(1j * k / (2 * distance_to_eye) * (X ** 2 + Y ** 2))
            field_at_eye = propagated_field * quadratic_phase
            # 使用菲涅耳传播方法
            field_at_eye = fresnel_propagation(field_at_eye, dx, dy, params.lbda, distance_to_eye)

            # 计算光强分布
            intensity = np.abs(field_at_eye) ** 2
            intensity_channel.append(intensity)

        # 合并通道
        intensity = np.stack(intensity_channel, axis=-1)
        intensities.append(intensity)

    # 对各次试验的强度取平均
    final_intensity = np.mean(intensities, axis=0)

    # 归一化强度以便显示
    final_intensity = final_intensity / np.max(final_intensity)

    return final_intensity

# 假设的裁剪函数，需要根据实际情况实现
def crop_grayscale_image(image, target_shape):
    return image[:target_shape[0], :target_shape[1]]

# 示例参数
class Params:
    oledPitch = 7e-6
    lbda = 530e-9

params = Params()
propagation_distance = 0.1  # 假设传播距离为0.1米

# 示例输入
texture_map = np.random.rand(100, 100, 3)  # 随机生成的纹理图
phase_mask = np.random.rand(100, 100)  # 随机生成的相位掩码

# 模拟光波传播
final_intensity = simulate_split_lohmann_system(texture_map, phase_mask, params, propagation_distance)

# 转换为图像格式并显示
final_intensity = final_intensity * 255
final_intensity = final_intensity.astype(np.uint8)

cv2.imshow('Eye View', final_intensity)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite('eye_view.png', final_intensity)