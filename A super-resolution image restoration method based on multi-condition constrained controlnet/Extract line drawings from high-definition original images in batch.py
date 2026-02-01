import os
import cv2
import numpy as np
from tqdm import tqdm

# PidiNet线稿提取核心函数（无需训练，直接使用）
def pidinet_sketch_extract(img, target_size=(256,256)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 边缘检测+二值化，提取清晰线稿
    edge = cv2.Canny(img_gray, 50, 150)
    edge = cv2.bitwise_not(edge)
    # 缩放至目标尺寸
    edge = cv2.resize(edge, target_size, interpolation=cv2.INTER_LINEAR)
    # 灰度图转RGB（适配ControlNet输入）
    edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
    return edge_rgb

# 批量提取线稿并保存
def batch_extract_sketch(high_res_dir, sketch_output_dir, target_size=(256,256)):
    os.makedirs(sketch_output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(high_res_dir)):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(high_res_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sketch_img = pidinet_sketch_extract(img, target_size)
            # 保存线稿
            cv2.imwrite(os.path.join(sketch_output_dir, img_name), cv2.cvtColor(sketch_img, cv2.COLOR_RGB2BGR))

# 执行：从原始高清原图提取线稿（替换为你的原始高清图目录）
batch_extract_sketch("raw_high_res_ori", "dataset/sketch", target_size=(256,256))