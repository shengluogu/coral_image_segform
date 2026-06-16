import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import math

def enhance_underwater_image(image):
    """
    完整版水下图像增强（基于物理模型+图像处理）
    
    参数:
    image: BGR格式的图像 (numpy array)
    
    返回:
    增强后的图像 (BGR格式)
    """
    # 步骤1: 水下颜色校正（补偿光谱衰减）
    corrected = color_correction(image)
    
    # 步骤2: 暗通道先验去雾（去除散射）
    # dehazed = dark_channel_prior_dehaze(corrected)
    
    # 步骤3: HSV空间CLAHE增强（提升对比度）
    enhanced = clahe_enhance(corrected)
    
    return enhanced

def color_correction(image):
    """
    基于水下光谱衰减模型的颜色校正
    """
    # 水下光谱衰减系数（经验参数，针对浅水珊瑚礁）
    k_blue = 0.025  # 蓝光衰减系数 (0.02-0.03)
    k_green = 0.015  # 绿光衰减系数 (0.01-0.02)
    k_red = 0.005    # 红光衰减系数 (0.004-0.006)
    
    # 转换为RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # 1. 计算光程距离（简化模型：基于图像深度估计）
    # 使用暗通道先验估计深度（简化版）
    dark_channel = np.min(rgb, axis=2)
    depth = 1.0 / (k_blue * np.log(0.95) + k_green * np.log(0.95) + k_red * np.log(0.95)) * np.log(1.0 / (dark_channel + 1e-6))
    
    # 2. 应用颜色校正（物理模型：I = J * exp(-k*d) + A*(1-exp(-k*d))）
    # 简化为：J = (I - A) / exp(-k*d) + A
    A = np.max(rgb, axis=2)  # 大气光
    #A = np.expand_dims(A, axis=2)
    
    # 对每个通道独立校正
    corrected = np.zeros_like(rgb)
    for c in range(3):
        k = [k_blue, k_green, k_red][c]
        # 校正公式：J_c = (I_c - A_c) * exp(k * depth) + A_c
        corrected[:, :, c] = (rgb[:, :, c] - A) * np.exp(k * depth) + A
    
    # 限制在0-255范围内
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

def dark_channel_prior_dehaze(image, omega=0.95, window_size=15):
    """
    暗通道先验去雾（水下增强专用）
    
    参数:
    image: BGR格式图像
    omega: 大气光比例参数 (0.9-0.95)
    window_size: 暗通道计算窗口大小
    """
    # 转换为RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 1. 计算暗通道
    dark_channel = np.min(rgb, axis=2)
    dark_channel = ndimage.minimum_filter(dark_channel, size=window_size)
    
    # 2. 估计大气光A
    A = np.max(rgb, axis=2)
    A = np.ones_like(dark_channel) * A
    
    # 3. 估计透射率
    t = 1.0 - omega * dark_channel
    
    # 4. 去雾公式：J = (I - A) / t + A
    dehazed = np.zeros_like(rgb)
    for c in range(3):
        dehazed[:, :, c] = (rgb[:, :, c] - A) / t + A
    
    # 限制在0-1范围
    dehazed = np.clip(dehazed, 0, 1)
    
    # 转换回BGR
    dehazed = (dehazed * 255).astype(np.uint8)
    return cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)

def clahe_enhance(image):
    """
    HSV空间CLAHE增强（水下图像专用）
    """
    # 转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 1. 对V通道应用CLAHE（增强对比度）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    
    # 2. 增强饱和度（水下图像通常饱和度低）
    s = cv2.convertScaleAbs(s, alpha=1.2, beta=0)  # 饱和度增强20%
    
    # 3. 调整色调（减少蓝色偏移）
    # 水下图像色调偏蓝（H≈120），我们向红色方向调整（H=0）
    h = np.clip(h - 15, 0, 179)  # H范围0-179
    
    # 4. 合并通道
    hsv_clahe = cv2.merge([h, s, v_clahe])
    
    # 转换回BGR
    enhanced_image = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    return enhanced_image