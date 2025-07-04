#TODO版本1.0（有一定可行性，功能不完善）
# 仅有最基础功能的模拟仿真调试
# import glob
# import os
# import numpy as np
# import skimage.io
# import cv2
# import copy
# import matplotlib.pyplot as plt
# from params import Params
# from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
#
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
#
# # 设置中文字体
# plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "KaiTi"]
#
# # 验证字体设置
# try:
#     font = fm.FontProperties(family="SimHei")
#     print(f"中文字体已设置为: {font.get_name()}")
# except:
#     print("警告: 未找到中文字体，中文可能无法正常显示")
#
# # 实现论文中从输入 RGBD 图像生成 OLED 显示纹理和 SLM 相位掩码的核心处理流程
#
# def digitize(diopter_map, numDepths):
#     '''
#     Discretize input diopter_map to numDepths different depths.
#     将输入的屈光度图离散化为 numDepths 个不同的深度。
#     '''
#     diopterBins = np.linspace(np.min(diopter_map), np.max(diopter_map), numDepths)
#     dig = np.digitize(diopter_map, diopterBins) - 1
#     return diopterBins[dig]
#
#
# def load_images(load_path, discretize=False, numdepths=None,
#                 texture_map_name=None, diopter_map_name=None):
#     '''
#     Load texture and diopter maps from separate pngs.
#     Assumes that texture map is color and the diopter map is grayscale.
#     从单独的 png 图像中加载纹理图和屈光度图。
#     假定纹理图为彩色图，屈光度图为灰度图。
#     '''
#     texture_map = skimage.io.imread(
#         os.path.join(load_path, texture_map_name)).astype(np.float64) / 255
#     diopter_map = skimage.io.imread(
#         os.path.join(load_path, diopter_map_name)).astype(np.float64) / 255
#     texture_map = texture_map[:, :, :3]
#
#     if discretize:
#         diopter_map = digitize(diopter_map, numdepths)
#
#     return texture_map, diopter_map
#
#
# def crop_grayscale_image(image, slm_shape):
#     '''
#     Assumes image is grayscale and has shape (h,w).
#     If image is bigger than slm_shape, crop image to fit slm_shape.
#     If image is smaller than slm_shape, pad images with zeros to fit slm_shape.
#     假设图像为灰度图，形状为 (h, w)。
#     如果图像比空间光调制器（SLM）的形状大，则裁剪图像以匹配 SLM 的形状。
#     如果图像比 SLM 的形状小，则用零填充图像以匹配 SLM 的形状。
#     '''
#     ## crop image to fit slm_shape
#     ystart, xstart = 0, 0
#     if image.shape[0] > slm_shape[0]:
#         ystart = (image.shape[0] - slm_shape[0]) // 2
#     if image.shape[1] > slm_shape[1]:
#         xstart = (image.shape[1] - slm_shape[1]) // 2
#     image = image[ystart:ystart + slm_shape[0], xstart:xstart + slm_shape[1]]
#
#     ## pad along y
#     if image.shape[0] < slm_shape[0]:
#         numpix2fill = slm_shape[0] - image.shape[0]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2 + 1), (0, 0)))
#         else:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2), (0, 0)))
#
#     ## pad along x
#     if image.shape[1] < slm_shape[1]:
#         numpix2fill = slm_shape[1] - image.shape[1]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2 + 1)))
#         else:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2)))
#
#     return image
#
#
# def crop_color_image(image, slm_shape):
#     '''
#     Assumes image is a color image and has shape (h,w,3).
#     If image is bigger than slm_shape, crop image to fit slm_shape.
#     If image is smaller than slm_shape, pad images with zeros to fit slm_shape.
#     假设图像是彩色图像，且形状为 (h, w, 3)。
#     如果图像大于空间光调制器（SLM）的形状，裁剪图像以适配 SLM 的形状。
#     如果图像小于 SLM 的形状，用零填充图像以适配 SLM 的形状。
#     '''
#     ## crop image to fit slm_shape
#     ystart, xstart = 0, 0
#     if image.shape[0] > slm_shape[0]:
#         ystart = (image.shape[0] - slm_shape[0]) // 2
#     if image.shape[1] > slm_shape[1]:
#         xstart = (image.shape[1] - slm_shape[1]) // 2
#     image = image[ystart:ystart + slm_shape[0], xstart:xstart + slm_shape[1], :]
#
#     ## pad along y
#     if image.shape[0] < slm_shape[0]:
#         numpix2fill = slm_shape[0] - image.shape[0]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2 + 1), (0, 0), (0, 0)))
#         else:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2), (0, 0), (0, 0)))
#
#     ## pad along x
#     if image.shape[1] < slm_shape[1]:
#         numpix2fill = slm_shape[1] - image.shape[1]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2 + 1), (0, 0)))
#         else:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2), (0, 0)))
#
#     return image
#
#
# def fit_images(H, texture_map, diopter_map, oled_shape, slm_shape):
#     '''
#     Crop and warp texture and depth maps.
#     For depth map, crop or pad it (no resizing) to fit the slm shape.
#     For texture map, crop or pad it (no resizing) to fit the slm shape then
#     backward warp it to fit on the oled.
#     裁剪并扭曲纹理图和深度图。
#     对于深度图，裁剪或填充它（不调整大小）以适配空间光调制器（SLM）的形状。
#     对于纹理图，裁剪或填充它（不调整大小）以适配空间光调制器（SLM）的形状，然后向后扭曲它以适配在有机发光二极管（OLED）上
#     '''
#     image_slm = np.flip(np.flip(diopter_map, 0), 1)
#     image_slm = crop_grayscale_image(image_slm, slm_shape)
#
#     image_oled = np.flip(np.flip(texture_map, 0), 1)
#     image_oled = crop_color_image(image_oled, slm_shape)
#     image_oled = cv2.warpPerspective(image_oled, np.linalg.inv(H), oled_shape)
#
#     return image_oled, image_slm
#
#
# def compute_phase_mask(diopterMap, params, name, modNum=1):
#     '''
#     Computes the phase mask from the normalized diopter map with the desired
#     working range.
#     根据所需的工作范围，从归一化屈光度图计算相位掩模。
#     '''
#     # fit diopter range
#     diopterMap = diopterMap * params.W
#
#     # define coordinates
#     xidx = np.arange(params.slmWidth) - params.slmWidth / 2
#     yidx = np.arange(params.slmHeight) - params.slmHeight / 2
#     X, Y = np.meshgrid(xidx, yidx)
#
#     # compute phase mask
#     factorY = (params.nominal_a / np.sqrt(1 + params.nominal_a ** 2))
#     scaleY = ((params.C0 * params.SLMpitch * params.fe ** 2) /
#               (3 * params.lbda * params.f0 ** 3)) * (params.W / 2 - diopterMap)
#     scaleX = scaleY / params.nominal_a
#     DeltaX = -scaleX * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#     DeltaY = -scaleY * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#     N = (params.lbda * params.f0) / params.SLMpitch
#
#     thetaX = DeltaX / N
#     thetaY = DeltaY / N
#     factor = modNum / (2 * np.pi)
#     phaseData = ((modNum * (thetaX * X + thetaY * Y)) % (modNum)) / factor
#     phaseData -= np.min(phaseData)
#     phaseData /= np.max(phaseData)
#
#     return phaseData
#
#
# def save2disk(path, device_name, img, name):
#     img = (np.round(img * 255)).astype(np.uint8)
#     os.mkdir(path) if not os.path.isdir(path) else None
#     savepath = os.path.join(path, device_name + '_' + name)
#     skimage.io.imsave(savepath, img)
#     print('Image saved to', savepath)
#
#
# # 角谱衍射理论模型实现
# def angular_spectrum_propagation(field, dx, dy, wavelength, distance):
#     """
#     使用角谱法模拟光波传播
#
#     参数:
#     field: 输入光场分布
#     dx, dy: 空间采样间隔(m)
#     wavelength: 波长(m)
#     distance: 传播距离(m)
#
#     返回:
#     propagated_field: 传播后的光场分布
#     """
#     # 输入光场的尺寸
#     ny, nx = field.shape
#
#     # 计算空间频率
#     fx = np.fft.fftfreq(nx, d=dx)
#     fy = np.fft.fftfreq(ny, d=dy)
#     FX, FY = np.meshgrid(fx, fy)
#
#     # 计算角谱相位因子
#     k = 2 * np.pi / wavelength  # 波数
#     phase_factor = np.exp(1j * k * distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))
#
#     # 处理数值计算中的奇点
#     phase_factor[np.isnan(phase_factor)] = 0
#
#     # 傅里叶变换到频域
#     field_freq = fft2(fftshift(field))
#
#     # 应用相位因子
#     field_freq_propagated = field_freq * phase_factor
#
#     # 逆傅里叶变换回空间域
#     propagated_field = ifftshift(ifft2(field_freq_propagated))
#
#     return propagated_field
#
#
# def simulate_split_lohmann_system(texture_map, phase_mask, params, propagation_distance):
#     """
#     模拟Split-Lohmann系统中的光波传播
#
#     参数:
#     texture_map: OLED纹理图
#     phase_mask: SLM相位掩码
#     params: 系统参数
#     propagation_distance: 从SLM到观察平面的距离(m)
#
#     返回:
#     intensity: 观察平面的光强分布
#     """
#     # 确保输入为单色光(取绿色通道)
#     monochrome_texture = np.dot(texture_map[..., :3], [0.2989, 0.5870, 0.1140])
#
#     # 调整monochrome_texture的形状与phase_mask一致
#     monochrome_texture = crop_grayscale_image(monochrome_texture, phase_mask.shape)
#
#     # 计算空间采样间隔
#     dx = params.oledPitch  # OLED像素间距作为采样间隔
#     dy = params.oledPitch
#
#     # 组合纹理和相位掩码(假设SLM对光场进行相位调制)
#     # 这里假设光从OLED出发，经过SLM相位调制
#     # 实际物理过程更复杂，此处为简化模型
#     field_at_slm = monochrome_texture * np.exp(1j * 2 * np.pi * phase_mask)
#
#     # 使用角谱法模拟从SLM到观察平面的传播
#     propagated_field = angular_spectrum_propagation(
#         field_at_slm, dx, dy, params.lbda, propagation_distance
#     )
#
#     # 计算光强分布
#     intensity = np.abs(propagated_field) ** 2
#
#     # 归一化强度以便显示
#     intensity = intensity / np.max(intensity)
#
#     return intensity
#
#
# if __name__ == '__main__':
#     ### 指定纹理和屈光度图像路径
#     data_folder = 'data'
#     output_folder = 'output'
#     texture_map_name = 'texture_map.png'
#     diopter_map_name = 'diopter_map.png'
#
#     ### 定义参数
#     discretize = True
#     numdepths = 50
#     params = Params()
#     params.W = 4.0
#     oled_shape = (1200, 1920)
#     slm_shape = (1080, 1920)
#     print('OLED目标图像形状:', oled_shape)
#     print('SLM目标图像形状:', slm_shape)
#
#     # 假设已存在变换矩阵
#     try:
#         H = np.load(os.path.join(data_folder, 'HomographyMatrixOLED2SLM.npy'))
#     except:
#         # 如果没有变换矩阵，创建一个恒等矩阵作为示例
#         H = np.eye(3)
#         print("使用恒等变换矩阵进行演示")
#
#     ### 执行计算
#     texture_map, diopter_map = load_images(
#         data_folder, discretize, numdepths, texture_map_name, diopter_map_name
#     )
#     texture_map_out, diopter_map_out = fit_images(
#         H, texture_map, diopter_map, oled_shape, slm_shape
#     )
#     phase_mask = compute_phase_mask(diopter_map_out, params, diopter_map_name)
#
#     ### 保存输出图像
#     save2disk(output_folder, 'OLED', texture_map_out, texture_map_name)
#     save2disk(output_folder, 'SLM', diopter_map_out, diopter_map_name)
#     save2disk(output_folder, 'SLM', phase_mask, 'phase_mask_' + diopter_map_name)
#
#     ### 角谱衍射仿真
#     print("开始角谱衍射仿真...")
#     # 定义仿真参数
#     propagation_distance = 0.2  # 传播距离20cm(可调整)
#
#     # 执行仿真
#     intensity = simulate_split_lohmann_system(
#         texture_map_out, phase_mask, params, propagation_distance
#     )
#
#     # 可视化第一张图片，包含OLED纹理图和SLM相位掩码
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(121)
#     plt.imshow(texture_map_out[..., :3])
#     plt.title('OLED纹理图')
#     plt.axis('off')
#
#     plt.subplot(122)
#     plt.imshow(phase_mask, cmap='viridis')
#     plt.title('SLM相位掩码')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, 'oled_slm_images.png'))
#
#     # 可视化第二张图片，包含传播距离处的光强分布和不同距离处的平均光强图
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(121)
#     plt.imshow(intensity, cmap='gray')
#     plt.title(f'传播距离 {propagation_distance * 100}cm 处的光强分布')
#     plt.axis('off')
#
#     # 模拟不同距离的聚焦效果
#     propagation_distances = [0.15, 0.2, 0.25]  # 不同传播距离
#     plt.subplot(122)
#     plt.plot(propagation_distances, [np.mean(intensity) for _ in propagation_distances], 'o-')
#     plt.title('不同距离处的平均光强')
#     plt.xlabel('传播距离(m)')
#     plt.ylabel('平均光强')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, 'diffraction_simulation.png'))
#     plt.show()
#
#     print("角谱衍射仿真完成，结果已保存")

#TODO版本2.0
# 根据实际物理模型做了修正

# import glob
# import os
# import numpy as np
# import skimage.io
# import cv2
# import copy
# import matplotlib.pyplot as plt
# from params import Params
# from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
#
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
#
# # 设置中文字体
# plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "KaiTi"]
#
# # 验证字体设置
# try:
#     font = fm.FontProperties(family="SimHei")
#     print(f"中文字体已设置为: {font.get_name()}")
# except:
#     print("警告: 未找到中文字体，中文可能无法正常显示")
#
# # 实现论文中从输入 RGBD 图像生成 OLED 显示纹理和 SLM 相位掩码的核心处理流程
#
# def digitize(diopter_map, numDepths):
#     '''
#     Discretize input diopter_map to numDepths different depths.
#     '''
#     diopterBins = np.linspace(np.min(diopter_map), np.max(diopter_map), numDepths)
#     dig = np.digitize(diopter_map, diopterBins) - 1
#     return diopterBins[dig]
#
#
# def load_images(load_path, discretize=False, numdepths=None,
#                 texture_map_name=None, diopter_map_name=None):
#     '''
#     Load texture and diopter maps from separate pngs.
#     Assumes that texture map is color and the diopter map is grayscale.
#     '''
#     texture_map = skimage.io.imread(
#         os.path.join(load_path, texture_map_name)).astype(np.float64) / 255
#     diopter_map = skimage.io.imread(
#         os.path.join(load_path, diopter_map_name)).astype(np.float64) / 255
#     texture_map = texture_map[:, :, :3]
#
#     if discretize:
#         diopter_map = digitize(diopter_map, numdepths)
#
#     return texture_map, diopter_map
#
#
# def crop_grayscale_image(image, slm_shape):
#     '''
#     Assumes image is grayscale and has shape (h,w).
#     If image is bigger than slm_shape, crop image to fit slm_shape.
#     If image is smaller than slm_shape, pad images with zeros to fit slm_shape.
#     '''
#     ## crop image to fit slm_shape
#     ystart, xstart = 0, 0
#     if image.shape[0] > slm_shape[0]:
#         ystart = (image.shape[0] - slm_shape[0]) // 2
#     if image.shape[1] > slm_shape[1]:
#         xstart = (image.shape[1] - slm_shape[1]) // 2
#     image = image[ystart:ystart + slm_shape[0], xstart:xstart + slm_shape[1]]
#
#     ## pad along y
#     if image.shape[0] < slm_shape[0]:
#         numpix2fill = slm_shape[0] - image.shape[0]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2 + 1), (0, 0)))
#         else:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2), (0, 0)))
#
#     ## pad along x
#     if image.shape[1] < slm_shape[1]:
#         numpix2fill = slm_shape[1] - image.shape[1]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2 + 1)))
#         else:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2)))
#
#     return image
#
#
# def crop_color_image(image, slm_shape):
#     '''
#     Assumes image is a color image and has shape (h,w,3).
#     If image is bigger than slm_shape, crop image to fit slm_shape.
#     If image is smaller than slm_shape, pad images with zeros to fit slm_shape.
#     '''
#     ## crop image to fit slm_shape
#     ystart, xstart = 0, 0
#     if image.shape[0] > slm_shape[0]:
#         ystart = (image.shape[0] - slm_shape[0]) // 2
#     if image.shape[1] > slm_shape[1]:
#         xstart = (image.shape[1] - slm_shape[1]) // 2
#     image = image[ystart:ystart + slm_shape[0], xstart:xstart + slm_shape[1], :]
#
#     ## pad along y
#     if image.shape[0] < slm_shape[0]:
#         numpix2fill = slm_shape[0] - image.shape[0]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2 + 1), (0, 0), (0, 0)))
#         else:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2), (0, 0), (0, 0)))
#
#     ## pad along x
#     if image.shape[1] < slm_shape[1]:
#         numpix2fill = slm_shape[1] - image.shape[1]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2 + 1), (0, 0)))
#         else:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2), (0, 0)))
#
#     return image
#
#
# def fit_images(H, texture_map, diopter_map, oled_shape, slm_shape):
#     '''
#     Crop and warp texture and depth maps.
#     For depth map, crop or pad it (no resizing) to fit the slm shape.
#     For texture map, crop or pad it (no resizing) to fit the slm shape then
#     backward warp it to fit on the oled.
#     '''
#     image_slm = np.flip(np.flip(diopter_map, 0), 1)
#     image_slm = crop_grayscale_image(image_slm, slm_shape)
#
#     image_oled = np.flip(np.flip(texture_map, 0), 1)
#     image_oled = crop_color_image(image_oled, slm_shape)
#     image_oled = cv2.warpPerspective(image_oled, np.linalg.inv(H), oled_shape)
#
#     return image_oled, image_slm
#
#
# def compute_phase_mask(diopterMap, params, name, modNum=1):
#     '''
#     Computes the phase mask from the normalized diopter map with the desired
#     working range.
#     '''
#     # fit diopter range
#     diopterMap = diopterMap * params.W
#
#     # define coordinates
#     xidx = np.arange(params.slmWidth) - params.slmWidth / 2
#     yidx = np.arange(params.slmHeight) - params.slmHeight / 2
#     X, Y = np.meshgrid(xidx, yidx)
#
#     # compute phase mask
#     factorY = (params.nominal_a / np.sqrt(1 + params.nominal_a ** 2))
#     scaleY = ((params.C0 * params.SLMpitch * params.fe ** 2) /
#               (3 * params.lbda * params.f0 ** 3)) * (params.W / 2 - diopterMap)
#     scaleX = scaleY / params.nominal_a
#     DeltaX = -scaleX * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#     DeltaY = -scaleY * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#     N = (params.lbda * params.f0) / params.SLMpitch
#
#     thetaX = DeltaX / N
#     thetaY = DeltaY / N
#     factor = modNum / (2 * np.pi)
#     phaseData = ((modNum * (thetaX * X + thetaY * Y)) % (modNum)) / factor
#     phaseData -= np.min(phaseData)
#     phaseData /= np.max(phaseData)
#
#     # 考虑横向偏移校正
#     # 这里简单假设横向偏移为一个固定值，实际需要根据论文公式进行计算
#     lateral_shift_x = 0.1
#     lateral_shift_y = 0.1
#     phaseData = np.roll(phaseData, (int(lateral_shift_y), int(lateral_shift_x)), axis=(0, 1))
#
#     return phaseData
#
#
# def save2disk(path, device_name, img, name):
#     img = (np.round(img * 255)).astype(np.uint8)
#     os.mkdir(path) if not os.path.isdir(path) else None
#     savepath = os.path.join(path, device_name + '_' + name)
#     skimage.io.imsave(savepath, img)
#     print('Image saved to', savepath)
#
#
# # 角谱衍射理论模型实现
# def angular_spectrum_propagation(field, dx, dy, wavelength, distance):
#     """
#     使用角谱法模拟光波传播
#
#     参数:
#     field: 输入光场分布
#     dx, dy: 空间采样间隔(m)
#     wavelength: 波长(m)
#     distance: 传播距离(m)
#
#     返回:
#     propagated_field: 传播后的光场分布
#     """
#     # 输入光场的尺寸
#     ny, nx = field.shape
#
#     # 计算空间频率
#     fx = np.fft.fftfreq(nx, d=dx)
#     fy = np.fft.fftfreq(ny, d=dy)
#     FX, FY = np.meshgrid(fx, fy)
#
#     # 计算角谱相位因子
#     k = 2 * np.pi / wavelength  # 波数
#     phase_factor = np.exp(1j * k * distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))
#
#     # 处理数值计算中的奇点
#     phase_factor[np.isnan(phase_factor)] = 0
#
#     # 傅里叶变换到频域
#     field_freq = fft2(fftshift(field))
#
#     # 应用相位因子
#     field_freq_propagated = field_freq * phase_factor
#
#     # 逆傅里叶变换回空间域
#     propagated_field = ifftshift(ifft2(field_freq_propagated))
#
#     return propagated_field
#
#
# # 显式模拟立方相位板 PSF
# def psf_cubic_phase_plate(params, dx, dy, nx, ny):
#     xidx = np.arange(nx) - nx / 2
#     yidx = np.arange(ny) - ny / 2
#     X, Y = np.meshgrid(xidx, yidx)
#
#     k = 2 * np.pi / params.lbda
#     phase = (params.C0 * (X * dx) ** 3 + params.C0 * (Y * dy) ** 3)
#     psf = np.exp(1j * k * phase)
#     return psf
#
#
# def simulate_split_lohmann_system(texture_map, phase_mask, params, propagation_distance, num_trials=10):
#     """
#     模拟Split-Lohmann系统中的光波传播
#
#     参数:
#     texture_map: OLED纹理图
#     phase_mask: SLM相位掩码
#     params: 系统参数
#     propagation_distance: 从SLM到观察平面的距离(m)
#     num_trials: 空间非相干性模拟的试验次数
#
#     返回:
#     intensity: 观察平面的光强分布
#     """
#     intensities = []
#
#     for _ in range(num_trials):
#         # 空间非相干性模拟：添加随机相位
#         random_phase = np.random.uniform(0, 2 * np.pi, phase_mask.shape)
#         phase_mask_with_random = phase_mask + random_phase / (2 * np.pi)
#
#         # 完善多色光处理：对每个通道进行处理
#         intensity_channel = []
#         for channel in range(texture_map.shape[2]):
#             monochrome_texture = texture_map[..., channel]
#
#             # 调整monochrome_texture的形状与phase_mask一致
#             monochrome_texture = crop_grayscale_image(monochrome_texture, phase_mask.shape)
#
#             # 计算空间采样间隔
#             dx = params.oledPitch  # OLED像素间距作为采样间隔
#             dy = params.oledPitch
#
#             # 组合纹理和相位掩码(假设SLM对光场进行相位调制)
#             # 这里假设光从OLED出发，经过SLM相位调制
#             # 实际物理过程更复杂，此处为简化模型
#             field_at_slm = monochrome_texture * np.exp(1j * 2 * np.pi * phase_mask_with_random)
#
#             # 显式模拟立方相位板 PSF
#             ny, nx = field_at_slm.shape
#             psf = psf_cubic_phase_plate(params, dx, dy, nx, ny)
#             field_at_slm = field_at_slm * psf
#
#             # 使用角谱法模拟从SLM到观察平面的传播
#             propagated_field = angular_spectrum_propagation(
#                 field_at_slm, dx, dy, params.lbda, propagation_distance
#             )
#
#             # 计算光强分布
#             intensity = np.abs(propagated_field) ** 2
#
#             # 归一化强度以便显示
#             intensity = intensity / np.max(intensity)
#
#             intensity_channel.append(intensity)
#
#         # 合并通道
#         intensity = np.stack(intensity_channel, axis=-1)
#         intensities.append(intensity)
#
#     # 多次平均
#     intensity = np.mean(intensities, axis=0)
#
#     return intensity
#
#
# if __name__ == '__main__':
#     ### 指定纹理和屈光度图像路径
#     data_folder = 'data'
#     output_folder = 'output'
#     texture_map_name = 'texture_map.png'
#     diopter_map_name = 'diopter_map.png'
#
#     ### 定义参数
#     discretize = True
#     numdepths = 50
#     params = Params()
#     params.W = 4.0
#     oled_shape = (1200, 1920)
#     slm_shape = (1080, 1920)
#     print('OLED目标图像形状:', oled_shape)
#     print('SLM目标图像形状:', slm_shape)
#
#     # 假设已存在变换矩阵
#     try:
#         H = np.load(os.path.join(data_folder, 'HomographyMatrixOLED2SLM.npy'))
#     except:
#         # 如果没有变换矩阵，创建一个恒等矩阵作为示例
#         H = np.eye(3)
#         print("使用恒等变换矩阵进行演示")
#
#     ### 执行计算
#     texture_map, diopter_map = load_images(
#         data_folder, discretize, numdepths, texture_map_name, diopter_map_name
#     )
#     texture_map_out, diopter_map_out = fit_images(
#         H, texture_map, diopter_map, oled_shape, slm_shape
#     )
#     phase_mask = compute_phase_mask(diopter_map_out, params, diopter_map_name)
#
#     ### 保存输出图像
#     save2disk(output_folder, 'OLED', texture_map_out, texture_map_name)
#     save2disk(output_folder, 'SLM', diopter_map_out, diopter_map_name)
#     save2disk(output_folder, 'SLM', phase_mask, 'phase_mask_' + diopter_map_name)
#
#     ### 角谱衍射仿真
#     print("开始角谱衍射仿真...")
#     # 定义仿真参数
#     propagation_distance = 0.2  # 传播距离20cm(可调整)
#     num_trials = 10  # 空间非相干性模拟的试验次数
#
#     # 执行仿真
#     intensity = simulate_split_lohmann_system(
#         texture_map_out, phase_mask, params, propagation_distance, num_trials
#     )
#
#     # 可视化第一张图片，包含OLED纹理图和SLM相位掩码
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(121)
#     plt.imshow(texture_map_out[..., :3])
#     plt.title('OLED纹理图')
#     plt.axis('off')
#
#     plt.subplot(122)
#     plt.imshow(phase_mask, cmap='viridis')
#     plt.title('SLM相位掩码')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, 'oled_slm_images.png'))
#
#     # 可视化第二张图片，包含传播距离处的光强分布和不同距离处的平均光强图
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(121)
#     plt.imshow(intensity, cmap='gray')
#     plt.title(f'传播距离 {propagation_distance * 100}cm 处的光强分布')
#     plt.axis('off')
#
#     # 模拟不同距离的聚焦效果
#     propagation_distances = [0.15, 0.2, 0.25]  # 不同传播距离
#     avg_intensities = []
#     for dist in propagation_distances:
#         intensity_dist = simulate_split_lohmann_system(
#             texture_map_out, phase_mask, params, dist, num_trials
#         )
#         avg_intensities.append(np.mean(intensity_dist))
#
#     plt.subplot(122)
#     plt.plot(propagation_distances, avg_intensities, 'o-')
#     plt.title('不同距离处的平均光强')
#     plt.xlabel('传播距离(m)')
#     plt.ylabel('平均光强')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, 'diffraction_simulation.png'))
#     plt.show()
#
#     print("角谱衍射仿真完成，结果已保存")

#TODO版本2.1（由豆包提供）
# 空间非相干性模拟：在 simulate_split_lohmann_system 函数中，通过 num_trials 参数控制试验次数，每次试验添加随机相位，最后对多次试验结果进行平均。
# 显式模拟立方相位板 PSF：添加了 psf_cubic_phase_plate 函数，用于计算立方相位板的点扩散函数，并在 simulate_split_lohmann_system 函数中应用。
# 完善多色光处理：在 simulate_split_lohmann_system 函数中，对纹理图的每个通道进行独立处理，最后合并通道。
# 横向偏移校正：在 compute_phase_mask 函数中，使用 np.roll 函数对相位掩码进行横向偏移。
# 原代码在每次模拟后对单个通道进行归一化，这可能导致整体图像偏暗。我可以修改代码，在合并所有通道后再进行全局归一化，这样可以增强图像的对比度。

# import glob
# import os
# import numpy as np
# import skimage.io
# import cv2
# import copy
# import matplotlib.pyplot as plt
# from params import Params
# from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
#
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
#
# # 设置中文字体
# plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "KaiTi"]
#
# # 验证字体设置
# try:
#     font = fm.FontProperties(family="SimHei")
#     print(f"中文字体已设置为: {font.get_name()}")
# except:
#     print("警告: 未找到中文字体，中文可能无法正常显示")
#
#
# # 实现论文中从输入 RGBD 图像生成 OLED 显示纹理和 SLM 相位掩码的核心处理流程
#
# def digitize(diopter_map, numDepths):
#     '''
#     Discretize input diopter_map to numDepths different depths.
#     '''
#     diopterBins = np.linspace(np.min(diopter_map), np.max(diopter_map), numDepths)
#     dig = np.digitize(diopter_map, diopterBins) - 1
#     return diopterBins[dig]
#
#
# def load_images(load_path, discretize=False, numdepths=None,
#                 texture_map_name=None, diopter_map_name=None):
#     '''
#     Load texture and diopter maps from separate pngs.
#     Assumes that texture map is color and the diopter map is grayscale.
#     '''
#     texture_map = skimage.io.imread(
#         os.path.join(load_path, texture_map_name)).astype(np.float64) / 255
#     diopter_map = skimage.io.imread(
#         os.path.join(load_path, diopter_map_name)).astype(np.float64) / 255
#     texture_map = texture_map[:, :, :3]
#
#     if discretize:
#         diopter_map = digitize(diopter_map, numdepths)
#
#     return texture_map, diopter_map
#
#
# def crop_grayscale_image(image, slm_shape):
#     '''
#     Assumes image is grayscale and has shape (h,w).
#     If image is bigger than slm_shape, crop image to fit slm_shape.
#     If image is smaller than slm_shape, pad images with zeros to fit slm_shape.
#     '''
#     ## crop image to fit slm_shape
#     ystart, xstart = 0, 0
#     if image.shape[0] > slm_shape[0]:
#         ystart = (image.shape[0] - slm_shape[0]) // 2
#     if image.shape[1] > slm_shape[1]:
#         xstart = (image.shape[1] - slm_shape[1]) // 2
#     image = image[ystart:ystart + slm_shape[0], xstart:xstart + slm_shape[1]]
#
#     ## pad along y
#     if image.shape[0] < slm_shape[0]:
#         numpix2fill = slm_shape[0] - image.shape[0]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2 + 1), (0, 0)))
#         else:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2), (0, 0)))
#
#     ## pad along x
#     if image.shape[1] < slm_shape[1]:
#         numpix2fill = slm_shape[1] - image.shape[1]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2 + 1)))
#         else:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2)))
#
#     return image
#
#
# def crop_color_image(image, slm_shape):
#     '''
#     Assumes image is a color image and has shape (h,w,3).
#     If image is bigger than slm_shape, crop image to fit slm_shape.
#     If image is smaller than slm_shape, pad images with zeros to fit slm_shape.
#     '''
#     ## crop image to fit slm_shape
#     ystart, xstart = 0, 0
#     if image.shape[0] > slm_shape[0]:
#         ystart = (image.shape[0] - slm_shape[0]) // 2
#     if image.shape[1] > slm_shape[1]:
#         xstart = (image.shape[1] - slm_shape[1]) // 2
#     image = image[ystart:ystart + slm_shape[0], xstart:xstart + slm_shape[1], :]
#
#     ## pad along y
#     if image.shape[0] < slm_shape[0]:
#         numpix2fill = slm_shape[0] - image.shape[0]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2 + 1), (0, 0), (0, 0)))
#         else:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2), (0, 0), (0, 0)))
#
#     ## pad along x
#     if image.shape[1] < slm_shape[1]:
#         numpix2fill = slm_shape[1] - image.shape[1]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2 + 1), (0, 0)))
#         else:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2), (0, 0)))
#
#     return image
#
#
# def fit_images(H, texture_map, diopter_map, oled_shape, slm_shape):
#     '''
#     Crop and warp texture and depth maps.
#     For depth map, crop or pad it (no resizing) to fit the slm shape.
#     For texture map, crop or pad it (no resizing) to fit the slm shape then
#     backward warp it to fit on the oled.
#     '''
#     image_slm = np.flip(np.flip(diopter_map, 0), 1)
#     image_slm = crop_grayscale_image(image_slm, slm_shape)
#
#     image_oled = np.flip(np.flip(texture_map, 0), 1)
#     image_oled = crop_color_image(image_oled, slm_shape)
#     image_oled = cv2.warpPerspective(image_oled, np.linalg.inv(H), oled_shape)
#
#     return image_oled, image_slm
#
#
# def compute_phase_mask(diopterMap, params, name, modNum=1):
#     '''
#     Computes the phase mask from the normalized diopter map with the desired
#     working range.
#     '''
#     # fit diopter range
#     diopterMap = diopterMap * params.W
#
#     # define coordinates
#     xidx = np.arange(params.slmWidth) - params.slmWidth / 2
#     yidx = np.arange(params.slmHeight) - params.slmHeight / 2
#     X, Y = np.meshgrid(xidx, yidx)
#
#     # compute phase mask
#     factorY = (params.nominal_a / np.sqrt(1 + params.nominal_a ** 2))
#     scaleY = ((params.C0 * params.SLMpitch * params.fe ** 2) /
#               (3 * params.lbda * params.f0 ** 3)) * (params.W / 2 - diopterMap)
#     scaleX = scaleY / params.nominal_a
#     DeltaX = -scaleX * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#     DeltaY = -scaleY * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#     N = (params.lbda * params.f0) / params.SLMpitch
#
#     thetaX = DeltaX / N
#     thetaY = DeltaY / N
#     factor = modNum / (2 * np.pi)
#     phaseData = ((modNum * (thetaX * X + thetaY * Y)) % (modNum)) / factor
#     phaseData -= np.min(phaseData)
#     phaseData /= np.max(phaseData)
#
#     # 考虑横向偏移校正
#     # 这里简单假设横向偏移为一个固定值，实际需要根据论文公式进行计算
#     lateral_shift_x = 0.1
#     lateral_shift_y = 0.1
#     phaseData = np.roll(phaseData, (int(lateral_shift_y), int(lateral_shift_x)), axis=(0, 1))
#
#     return phaseData
#
#
# def save2disk(path, device_name, img, name):
#     img = (np.round(img * 255)).astype(np.uint8)
#     os.mkdir(path) if not os.path.isdir(path) else None
#     savepath = os.path.join(path, device_name + '_' + name)
#     skimage.io.imsave(savepath, img)
#     print('Image saved to', savepath)
#
#
# # 角谱衍射理论模型实现
# def angular_spectrum_propagation(field, dx, dy, wavelength, distance):
#     """
#     使用角谱法模拟光波传播
#
#     参数:
#     field: 输入光场分布
#     dx, dy: 空间采样间隔(m)
#     wavelength: 波长(m)
#     distance: 传播距离(m)
#
#     返回:
#     propagated_field: 传播后的光场分布
#     """
#     # 输入光场的尺寸
#     ny, nx = field.shape
#
#     # 计算空间频率
#     fx = np.fft.fftfreq(nx, d=dx)
#     fy = np.fft.fftfreq(ny, d=dy)
#     FX, FY = np.meshgrid(fx, fy)
#
#     # 计算角谱相位因子
#     k = 2 * np.pi / wavelength  # 波数
#     phase_factor = np.exp(1j * k * distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))
#
#     # 处理数值计算中的奇点
#     phase_factor[np.isnan(phase_factor)] = 0
#
#     # 傅里叶变换到频域
#     field_freq = fft2(fftshift(field))
#
#     # 应用相位因子
#     field_freq_propagated = field_freq * phase_factor
#
#     # 逆傅里叶变换回空间域
#     propagated_field = ifftshift(ifft2(field_freq_propagated))
#
#     return propagated_field
#
#
# # 显式模拟立方相位板 PSF
# def psf_cubic_phase_plate(params, dx, dy, nx, ny):
#     xidx = np.arange(nx) - nx / 2
#     yidx = np.arange(ny) - ny / 2
#     X, Y = np.meshgrid(xidx, yidx)
#
#     k = 2 * np.pi / params.lbda
#     phase = (params.C0 * (X * dx) ** 3 + params.C0 * (Y * dy) ** 3)
#     psf = np.exp(1j * k * phase)
#     return psf
#
#
# def simulate_split_lohmann_system(texture_map, phase_mask, params, propagation_distance, num_trials=10):
#     """
#     模拟Split-Lohmann系统中的光波传播
#
#     参数:
#     texture_map: OLED纹理图
#     phase_mask: SLM相位掩码
#     params: 系统参数
#     propagation_distance: 从SLM到观察平面的距离(m)
#     num_trials: 空间非相干性模拟的试验次数
#
#     返回:
#     intensity: 观察平面的光强分布
#     """
#     intensities = []
#
#     for _ in range(num_trials):
#         # 空间非相干性模拟：添加随机相位
#         random_phase = np.random.uniform(0, 2 * np.pi, phase_mask.shape)
#         phase_mask_with_random = phase_mask + random_phase / (2 * np.pi)
#
#         # 完善多色光处理：对每个通道进行处理
#         intensity_channel = []
#         for channel in range(texture_map.shape[2]):
#             monochrome_texture = texture_map[..., channel]
#
#             # 调整monochrome_texture的形状与phase_mask一致
#             monochrome_texture = crop_grayscale_image(monochrome_texture, phase_mask.shape)
#
#             # 计算空间采样间隔
#             dx = params.oledPitch  # OLED像素间距作为采样间隔
#             dy = params.oledPitch
#
#             # 组合纹理和相位掩码(假设SLM对光场进行相位调制)
#             # 这里假设光从OLED出发，经过SLM相位调制
#             # 实际物理过程更复杂，此处为简化模型
#             field_at_slm = monochrome_texture * np.exp(1j * 2 * np.pi * phase_mask_with_random)
#
#             # 显式模拟立方相位板 PSF
#             ny, nx = field_at_slm.shape
#             psf = psf_cubic_phase_plate(params, dx, dy, nx, ny)
#             field_at_slm = field_at_slm * psf
#
#             # 使用角谱法模拟从SLM到观察平面的传播
#             propagated_field = angular_spectrum_propagation(
#                 field_at_slm, dx, dy, params.lbda, propagation_distance
#             )
#
#             # 计算光强分布
#             intensity = np.abs(propagated_field) ** 2
#
#             # 暂时不进行通道归一化，改为全局归一化
#             intensity_channel.append(intensity)
#
#         # 合并通道
#         intensity = np.stack(intensity_channel, axis=-1)
#         intensities.append(intensity)
#
#     # 多次平均
#     intensity = np.mean(intensities, axis=0)
#
#     # 全局归一化，提高对比度
#     intensity = intensity - np.min(intensity)
#     intensity = intensity / np.max(intensity)
#
#     return intensity
#
#
# if __name__ == '__main__':
#     ### 指定纹理和屈光度图像路径
#     data_folder = 'data'
#     output_folder = 'output'
#     texture_map_name = 'texture_map.png'
#     diopter_map_name = 'diopter_map.png'
#
#     ### 定义参数
#     discretize = True
#     numdepths = 50
#     params = Params()
#     params.W = 4.0
#     oled_shape = (1200, 1920)
#     slm_shape = (1080, 1920)
#     print('OLED目标图像形状:', oled_shape)
#     print('SLM目标图像形状:', slm_shape)
#
#     # 假设已存在变换矩阵
#     try:
#         H = np.load(os.path.join(data_folder, 'HomographyMatrixOLED2SLM.npy'))
#     except:
#         # 如果没有变换矩阵，创建一个恒等矩阵作为示例
#         H = np.eye(3)
#         print("使用恒等变换矩阵进行演示")
#
#     ### 执行计算
#     texture_map, diopter_map = load_images(
#         data_folder, discretize, numdepths, texture_map_name, diopter_map_name
#     )
#     texture_map_out, diopter_map_out = fit_images(
#         H, texture_map, diopter_map, oled_shape, slm_shape
#     )
#     phase_mask = compute_phase_mask(diopter_map_out, params, diopter_map_name)
#
#     ### 保存输出图像
#     save2disk(output_folder, 'OLED', texture_map_out, texture_map_name)
#     save2disk(output_folder, 'SLM', diopter_map_out, diopter_map_name)
#     save2disk(output_folder, 'SLM', phase_mask, 'phase_mask_' + diopter_map_name)
#
#     ### 角谱衍射仿真
#     print("开始角谱衍射仿真...")
#     # 定义仿真参数
#     propagation_distance = 0.2  # 传播距离20cm(可调整)
#     num_trials = 10  # 空间非相干性模拟的试验次数
#
#     # 执行仿真
#     intensity = simulate_split_lohmann_system(
#         texture_map_out, phase_mask, params, propagation_distance, num_trials
#     )
#
#     # 可视化第一张图片，包含OLED纹理图和SLM相位掩码
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(121)
#     plt.imshow(texture_map_out[..., :3])
#     plt.title('OLED纹理图')
#     plt.axis('off')
#
#     plt.subplot(122)
#     plt.imshow(phase_mask, cmap='viridis')
#     plt.title('SLM相位掩码')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, 'oled_slm_images.png'))
#
#     # 可视化第二张图片，包含传播距离处的光强分布和不同距离处的平均光强图
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(121)
#     # 使用'jet'或其他彩色cmap代替'gray'，增强可视化效果
#     plt.imshow(intensity, cmap='jet')
#     plt.title(f'传播距离 {propagation_distance * 100}cm 处的光强分布')
#     plt.axis('off')
#
#     # 模拟不同距离的聚焦效果
#     propagation_distances = [0.15, 0.2, 0.25]  # 不同传播距离
#     avg_intensities = []
#     for dist in propagation_distances:
#         intensity_dist = simulate_split_lohmann_system(
#             texture_map_out, phase_mask, params, dist, num_trials
#         )
#         avg_intensities.append(np.mean(intensity_dist))
#
#     plt.subplot(122)
#     plt.plot(propagation_distances, avg_intensities, 'o-')
#     plt.title('不同距离处的平均光强')
#     plt.xlabel('传播距离(m)')
#     plt.ylabel('平均光强')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, 'diffraction_simulation.png'))
#     plt.show()
#
#     print("角谱衍射仿真完成，结果已保存")

#TODO版本2.2（由豆包提供，已验证不可行）
# 对横向偏移校正部分进行修改，我们需要根据原论文中的公式17来计算横向偏移量。

# import glob
# import os
# import numpy as np
# import skimage.io
# import cv2
# import copy
# import matplotlib.pyplot as plt
# from params import Params
# from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
#
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
#
# # 设置中文字体
# plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "KaiTi"]
#
# # 验证字体设置
# try:
#     font = fm.FontProperties(family="SimHei")
#     print(f"中文字体已设置为: {font.get_name()}")
# except:
#     print("警告: 未找到中文字体，中文可能无法正常显示")
#
#
# # 实现论文中从输入 RGBD 图像生成 OLED 显示纹理和 SLM 相位掩码的核心处理流程
#
# def digitize(diopter_map, numDepths):
#     '''
#     Discretize input diopter_map to numDepths different depths.
#     '''
#     diopterBins = np.linspace(np.min(diopter_map), np.max(diopter_map), numDepths)
#     dig = np.digitize(diopter_map, diopterBins) - 1
#     return diopterBins[dig]
#
#
# def load_images(load_path, discretize=False, numdepths=None,
#                 texture_map_name=None, diopter_map_name=None):
#     '''
#     Load texture and diopter maps from separate pngs.
#     Assumes that texture map is color and the diopter map is grayscale.
#     '''
#     texture_map = skimage.io.imread(
#         os.path.join(load_path, texture_map_name)).astype(np.float64) / 255
#     diopter_map = skimage.io.imread(
#         os.path.join(load_path, diopter_map_name)).astype(np.float64) / 255
#     texture_map = texture_map[:, :, :3]
#
#     if discretize:
#         diopter_map = digitize(diopter_map, numdepths)
#
#     return texture_map, diopter_map
#
#
# def crop_grayscale_image(image, slm_shape):
#     '''
#     Assumes image is grayscale and has shape (h,w).
#     If image is bigger than slm_shape, crop image to fit slm_shape.
#     If image is smaller than slm_shape, pad images with zeros to fit slm_shape.
#     '''
#     ## crop image to fit slm_shape
#     ystart, xstart = 0, 0
#     if image.shape[0] > slm_shape[0]:
#         ystart = (image.shape[0] - slm_shape[0]) // 2
#     if image.shape[1] > slm_shape[1]:
#         xstart = (image.shape[1] - slm_shape[1]) // 2
#     image = image[ystart:ystart + slm_shape[0], xstart:xstart + slm_shape[1]]
#
#     ## pad along y
#     if image.shape[0] < slm_shape[0]:
#         numpix2fill = slm_shape[0] - image.shape[0]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2 + 1), (0, 0)))
#         else:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2), (0, 0)))
#
#     ## pad along x
#     if image.shape[1] < slm_shape[1]:
#         numpix2fill = slm_shape[1] - image.shape[1]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2 + 1)))
#         else:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2)))
#
#     return image
#
#
# def crop_color_image(image, slm_shape):
#     '''
#     Assumes image is a color image and has shape (h,w,3).
#     If image is bigger than slm_shape, crop image to fit slm_shape.
#     If image is smaller than slm_shape, pad images with zeros to fit slm_shape.
#     '''
#     ## crop image to fit slm_shape
#     ystart, xstart = 0, 0
#     if image.shape[0] > slm_shape[0]:
#         ystart = (image.shape[0] - slm_shape[0]) // 2
#     if image.shape[1] > slm_shape[1]:
#         xstart = (image.shape[1] - slm_shape[1]) // 2
#     image = image[ystart:ystart + slm_shape[0], xstart:xstart + slm_shape[1], :]
#
#     ## pad along y
#     if image.shape[0] < slm_shape[0]:
#         numpix2fill = slm_shape[0] - image.shape[0]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2 + 1), (0, 0), (0, 0)))
#         else:
#             image = np.pad(image, ((numpix2fill // 2, numpix2fill // 2), (0, 0), (0, 0)))
#
#     ## pad along x
#     if image.shape[1] < slm_shape[1]:
#         numpix2fill = slm_shape[1] - image.shape[1]
#         if numpix2fill % 2 == 1:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2 + 1), (0, 0)))
#         else:
#             image = np.pad(image, ((0, 0), (numpix2fill // 2, numpix2fill // 2), (0, 0)))
#
#     return image
#
#
# def fit_images(H, texture_map, diopter_map, oled_shape, slm_shape):
#     '''
#     Crop and warp texture and depth maps.
#     For depth map, crop or pad it (no resizing) to fit the slm shape.
#     For texture map, crop or pad it (no resizing) to fit the slm shape then
#     backward warp it to fit on the oled.
#     '''
#     image_slm = np.flip(np.flip(diopter_map, 0), 1)
#     image_slm = crop_grayscale_image(image_slm, slm_shape)
#
#     image_oled = np.flip(np.flip(texture_map, 0), 1)
#     image_oled = crop_color_image(image_oled, slm_shape)
#     image_oled = cv2.warpPerspective(image_oled, np.linalg.inv(H), oled_shape)
#
#     return image_oled, image_slm
#
# #FIXME
# # def compute_phase_mask(diopterMap, params, name, modNum=1):
# #     '''
# #     Computes the phase mask from the normalized diopter map with the desired
# #     working range.
# #     '''
# #     # fit diopter range
# #     diopterMap = diopterMap * params.W
# #
# #     # define coordinates
# #     xidx = np.arange(params.slmWidth) - params.slmWidth / 2
# #     yidx = np.arange(params.slmHeight) - params.slmHeight / 2
# #     X, Y = np.meshgrid(xidx, yidx)
# #
# #     # compute phase mask
# #     factorY = (params.nominal_a / np.sqrt(1 + params.nominal_a ** 2))
# #     scaleY = ((params.C0 * params.SLMpitch * params.fe ** 2) /
# #               (3 * params.lbda * params.f0 ** 3)) * (params.W / 2 - diopterMap)
# #     scaleX = scaleY / params.nominal_a
# #     DeltaX = -scaleX * ((params.lbda * params.f0) / (2 * params.SLMpitch))
# #     DeltaY = -scaleY * ((params.lbda * params.f0) / (2 * params.SLMpitch))
# #     N = (params.lbda * params.f0) / params.SLMpitch
# #
# #     thetaX = DeltaX / N
# #     thetaY = DeltaY / N
# #     factor = modNum / (2 * np.pi)
# #     phaseData = ((modNum * (thetaX * X + thetaY * Y)) % (modNum)) / factor
# #     phaseData -= np.min(phaseData)
# #     phaseData /= np.max(phaseData)
# #
# #     # 考虑横向偏移校正
# #     # # 这里简单假设横向偏移为一个固定值，实际需要根据论文公式进行计算
# #     # lateral_shift_x = 0.1
# #     # lateral_shift_y = 0.1
# #     # phaseData = np.roll(phaseData, (int(lateral_shift_y), int(lateral_shift_x)), axis=(0, 1))
# #
# #     # 根据论文公式计算横向偏移量
# #     v0 = np.abs(DeltaX * N / (params.lbda * params.f0))  # 这里简单假设 v0 与 DeltaX 相关，实际可能需要更准确的计算
# #     lateral_shift_x = 3 * v0 ** 2 * params.f0 ** 3 / params.C0
# #     lateral_shift_y = 3 * v0 ** 2 * params.f0 ** 3 / params.C0
# #     # 将偏移量转换为单个数值
# #     lateral_shift_x = int(np.mean(lateral_shift_x))
# #     lateral_shift_y = int(np.mean(lateral_shift_y))
# #
# #     phaseData = np.roll(phaseData, (int(lateral_shift_y), int(lateral_shift_x)), axis=(0, 1))
# #
# #     return phaseData
#
# def compute_phase_mask(diopterMap, params, name, modNum=1):
#     '''
#     Computes the phase mask from the normalized diopter map with the desired
#     working range.
#     '''
#     # fit diopter range
#     diopterMap = diopterMap * params.W
#
#     # define coordinates
#     xidx = np.arange(params.slmWidth) - params.slmWidth / 2
#     yidx = np.arange(params.slmHeight) - params.slmHeight / 2
#     X, Y = np.meshgrid(xidx, yidx)
#
#     # compute phase mask
#     factorY = (params.nominal_a / np.sqrt(1 + params.nominal_a ** 2))
#     scaleY = ((params.C0 * params.SLMpitch * params.fe ** 2) /
#               (3 * params.lbda * params.f0 ** 3)) * (params.W / 2 - diopterMap)
#     scaleX = scaleY / params.nominal_a
#     DeltaX = -scaleX * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#     DeltaY = -scaleY * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#     N = (params.lbda * params.f0) / params.SLMpitch
#
#     thetaX = DeltaX / N
#     thetaY = DeltaY / N
#     factor = modNum / (2 * np.pi)
#     phaseData = ((modNum * (thetaX * X + thetaY * Y)) % (modNum)) / factor
#     phaseData -= np.min(phaseData)
#     phaseData /= np.max(phaseData)
#
#     # 根据论文公式计算横向偏移
#     lateral_shift_x = calculate_lateral_shift(diopterMap, params, axis='x')
#     lateral_shift_y = calculate_lateral_shift(diopterMap, params, axis='y')
#
#     # 使用更精确的空间变换替代简单的平移
#     phaseData = apply_spatial_shift(phaseData, lateral_shift_x, lateral_shift_y)
#
#     return phaseData
#
#
# def calculate_lateral_shift(diopterMap, params, axis='x'):
#     """
#     根据论文公式计算横向偏移量
#
#     参数:
#     diopterMap: 屈光度图
#     params: 系统参数
#     axis: 'x' 或 'y'，指定计算哪个方向的偏移
#
#     返回:
#     lateral_shift: 横向偏移量数组
#     """
#     # 计算 v0
#     factorY = (params.nominal_a / np.sqrt(1 + params.nominal_a ** 2))
#     scaleY = ((params.C0 * params.SLMpitch * params.fe ** 2) /
#               (3 * params.lbda * params.f0 ** 3)) * (params.W / 2 - diopterMap)
#
#     if axis == 'x':
#         scale = scaleY / params.nominal_a
#         Delta = -scale * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#     else:  # y 方向
#         Delta = -scaleY * ((params.lbda * params.f0) / (2 * params.SLMpitch))
#
#     N = (params.lbda * params.f0) / params.SLMpitch
#     v0 = np.abs(Delta * N / (params.lbda * params.f0))
#
#     # 根据论文公式计算横向偏移
#     lateral_shift = 3 * v0 ** 2 * params.f0 ** 3 / params.C0
#
#     # 转换为像素单位
#     pixel_shift = lateral_shift / params.SLMpitch
#
#     return pixel_shift
#
#
# def apply_spatial_shift(image, shift_x, shift_y):
#     """
#     应用空间变换，替代简单的平移
#
#     参数:
#     image: 输入图像
#     shift_x: x 方向的偏移量数组
#     shift_y: y 方向的偏移量数组
#
#     返回:
#     shifted_image: 经过变换的图像
#     """
#     # 创建网格
#     rows, cols = image.shape[:2]
#     x, y = np.meshgrid(np.arange(cols), np.arange(rows))
#
#     # 计算新的坐标
#     new_x = (x + shift_x).astype(np.float32)
#     new_y = (y + shift_y).astype(np.float32)
#
#     # 使用OpenCV的remap函数进行重映射
#     if len(image.shape) == 3:  # 彩色图像
#         channels = [cv2.remap(image[:, :, i], new_x, new_y, cv2.INTER_LINEAR) for i in range(3)]
#         shifted_image = np.stack(channels, axis=2)
#     else:  # 灰度图像
#         shifted_image = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR)
#
#     return shifted_image
#
#
# def save2disk(path, device_name, img, name):
#     img = (np.round(img * 255)).astype(np.uint8)
#     os.mkdir(path) if not os.path.isdir(path) else None
#     savepath = os.path.join(path, device_name + '_' + name)
#     skimage.io.imsave(savepath, img)
#     print('Image saved to', savepath)
#
#
# # 角谱衍射理论模型实现
# def angular_spectrum_propagation(field, dx, dy, wavelength, distance):
#     """
#     使用角谱法模拟光波传播
#
#     参数:
#     field: 输入光场分布
#     dx, dy: 空间采样间隔(m)
#     wavelength: 波长(m)
#     distance: 传播距离(m)
#
#     返回:
#     propagated_field: 传播后的光场分布
#     """
#     # 输入光场的尺寸
#     ny, nx = field.shape
#
#     # 计算空间频率
#     fx = np.fft.fftfreq(nx, d=dx)
#     fy = np.fft.fftfreq(ny, d=dy)
#     FX, FY = np.meshgrid(fx, fy)
#
#     # 计算角谱相位因子
#     k = 2 * np.pi / wavelength  # 波数
#     phase_factor = np.exp(1j * k * distance * np.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2))
#
#     # 处理数值计算中的奇点
#     phase_factor[np.isnan(phase_factor)] = 0
#
#     # 傅里叶变换到频域
#     field_freq = fft2(fftshift(field))
#
#     # 应用相位因子
#     field_freq_propagated = field_freq * phase_factor
#
#     # 逆傅里叶变换回空间域
#     propagated_field = ifftshift(ifft2(field_freq_propagated))
#
#     return propagated_field
#
#
# # 显式模拟立方相位板 PSF
# def psf_cubic_phase_plate(params, dx, dy, nx, ny):
#     xidx = np.arange(nx) - nx / 2
#     yidx = np.arange(ny) - ny / 2
#     X, Y = np.meshgrid(xidx, yidx)
#
#     k = 2 * np.pi / params.lbda
#     phase = (params.C0 * (X * dx) ** 3 + params.C0 * (Y * dy) ** 3)
#     psf = np.exp(1j * k * phase)
#     return psf
#
#
# def simulate_split_lohmann_system(texture_map, phase_mask, params, propagation_distance, num_trials=10):
#     """
#     模拟Split-Lohmann系统中的光波传播
#
#     参数:
#     texture_map: OLED纹理图
#     phase_mask: SLM相位掩码
#     params: 系统参数
#     propagation_distance: 从SLM到观察平面的距离(m)
#     num_trials: 空间非相干性模拟的试验次数
#
#     返回:
#     intensity: 观察平面的光强分布
#     """
#     intensities = []
#
#     for _ in range(num_trials):
#         # 空间非相干性模拟：添加随机相位
#         random_phase = np.random.uniform(0, 2 * np.pi, phase_mask.shape)
#         phase_mask_with_random = phase_mask + random_phase / (2 * np.pi)
#
#         # 完善多色光处理：对每个通道进行处理
#         intensity_channel = []
#         for channel in range(texture_map.shape[2]):
#             monochrome_texture = texture_map[..., channel]
#
#             # 调整monochrome_texture的形状与phase_mask一致
#             monochrome_texture = crop_grayscale_image(monochrome_texture, phase_mask.shape)
#
#             # 计算空间采样间隔
#             dx = params.oledPitch  # OLED像素间距作为采样间隔
#             dy = params.oledPitch
#
#             # 组合纹理和相位掩码(假设SLM对光场进行相位调制)
#             # 这里假设光从OLED出发，经过SLM相位调制
#             # 实际物理过程更复杂，此处为简化模型
#             field_at_slm = monochrome_texture * np.exp(1j * 2 * np.pi * phase_mask_with_random)
#
#             # 显式模拟立方相位板 PSF
#             ny, nx = field_at_slm.shape
#             psf = psf_cubic_phase_plate(params, dx, dy, nx, ny)
#             field_at_slm = field_at_slm * psf
#
#             # 使用角谱法模拟从SLM到观察平面的传播
#             propagated_field = angular_spectrum_propagation(
#                 field_at_slm, dx, dy, params.lbda, propagation_distance
#             )
#
#             # 计算光强分布
#             intensity = np.abs(propagated_field) ** 2
#
#             # 暂时不进行通道归一化，改为全局归一化
#             intensity_channel.append(intensity)
#
#         # 合并通道
#         intensity = np.stack(intensity_channel, axis=-1)
#         intensities.append(intensity)
#
#     # 多次平均
#     intensity = np.mean(intensities, axis=0)
#
#     # 全局归一化，提高对比度
#     intensity = intensity - np.min(intensity)
#     intensity = intensity / np.max(intensity)
#
#     return intensity
#
#
# if __name__ == '__main__':
#     ### 指定纹理和屈光度图像路径
#     data_folder = 'data'
#     output_folder = 'output'
#     texture_map_name = 'texture_map.png'
#     diopter_map_name = 'diopter_map.png'
#
#     ### 定义参数
#     discretize = True
#     numdepths = 50
#     params = Params()
#     params.W = 4.0
#     oled_shape = (1200, 1920)
#     slm_shape = (1080, 1920)
#     print('OLED目标图像形状:', oled_shape)
#     print('SLM目标图像形状:', slm_shape)
#
#     # 假设已存在变换矩阵
#     try:
#         H = np.load(os.path.join(data_folder, 'HomographyMatrixOLED2SLM.npy'))
#     except:
#         # 如果没有变换矩阵，创建一个恒等矩阵作为示例
#         H = np.eye(3)
#         print("使用恒等变换矩阵进行演示")
#
#     ### 执行计算
#     texture_map, diopter_map = load_images(
#         data_folder, discretize, numdepths, texture_map_name, diopter_map_name
#     )
#     texture_map_out, diopter_map_out = fit_images(
#         H, texture_map, diopter_map, oled_shape, slm_shape
#     )
#     phase_mask = compute_phase_mask(diopter_map_out, params, diopter_map_name)
#
#     ### 保存输出图像
#     save2disk(output_folder, 'OLED', texture_map_out, texture_map_name)
#     save2disk(output_folder, 'SLM', diopter_map_out, diopter_map_name)
#     save2disk(output_folder, 'SLM', phase_mask, 'phase_mask_' + diopter_map_name)
#
#     ### 角谱衍射仿真
#     print("开始角谱衍射仿真...")
#     # 定义仿真参数
#     propagation_distance = 0.2  # 传播距离20cm(可调整)
#     num_trials = 10  # 空间非相干性模拟的试验次数
#
#     # 执行仿真
#     intensity = simulate_split_lohmann_system(
#         texture_map_out, phase_mask, params, propagation_distance, num_trials
#     )
#
#     # 可视化第一张图片，包含OLED纹理图和SLM相位掩码
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(121)
#     plt.imshow(texture_map_out[..., :3])
#     plt.title('OLED纹理图')
#     plt.axis('off')
#
#     plt.subplot(122)
#     plt.imshow(phase_mask, cmap='viridis')
#     plt.title('SLM相位掩码')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, 'oled_slm_images.png'))
#
#     # 可视化第二张图片，包含传播距离处的光强分布和不同距离处的平均光强图
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(121)
#     # 使用'jet'或其他彩色cmap代替'gray'，增强可视化效果
#     plt.imshow(intensity, cmap='jet')
#     plt.title(f'传播距离 {propagation_distance * 100}cm 处的光强分布')
#     plt.axis('off')
#
#     # 模拟不同距离的聚焦效果
#     propagation_distances = [0.15, 0.2, 0.25]  # 不同传播距离
#     avg_intensities = []
#     for dist in propagation_distances:
#         intensity_dist = simulate_split_lohmann_system(
#             texture_map_out, phase_mask, params, dist, num_trials
#         )
#         avg_intensities.append(np.mean(intensity_dist))
#
#     plt.subplot(122)
#     plt.plot(propagation_distances, avg_intensities, 'o-')
#     plt.title('不同距离处的平均光强')
#     plt.xlabel('传播距离(m)')
#     plt.ylabel('平均光强')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, 'diffraction_simulation.png'))
#     plt.show()
#
#     print("角谱衍射仿真完成，结果已保存")