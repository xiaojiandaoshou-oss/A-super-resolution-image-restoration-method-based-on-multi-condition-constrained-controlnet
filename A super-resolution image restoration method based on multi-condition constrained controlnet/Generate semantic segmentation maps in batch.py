# 批量缩放生成低分辨率原图/高分辨率原图/分割图
import os
import cv2
from tqdm import tqdm

def batch_resize(img_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(img_dir)):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(output_dir, img_name), img_resized)

# 用法示例：
# 1. 从原始高清原图生成512×512的目标图
batch_resize("raw_high_res", "dataset/high_res_ori", (512, 512))
# 2. 从512×512目标图生成256×256的低分辨率原图
batch_resize("dataset/high_res_ori", "dataset/low_res_ori", (256, 256))
# 3. 从原始分割图生成256×256的适配分割图
batch_resize("raw_seg_mask", "dataset/seg_mask", (256, 256))
