import os
import cv2
from tqdm import tqdm

# 批量缩放图片
def batch_resize_ori(high_res_dir, low_res_output_dir, target_size=(256,256)):
    os.makedirs(low_res_output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(high_res_dir)):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(high_res_dir, img_name)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(low_res_output_dir, img_name), img_resized)

# 执行：生成低分辨率原图
batch_resize_ori("dataset/high_res_ori", "dataset/low_res_ori", target_size=(256,256))
# 可选：将原始高清图缩放到512×512（统一目标图尺寸）
batch_resize_ori("raw_high_res_ori", "dataset/high_res_ori", target_size=(512,512))