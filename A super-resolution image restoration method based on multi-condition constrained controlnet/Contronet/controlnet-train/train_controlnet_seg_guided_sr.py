import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffusers import ControlNetModel, UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===================== 核心配置参数（根据显卡/数据集修改）=====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5"  # SD1.5，自动从HF下载
# 三线配对数据集路径（必须文件名一一对应）
LOW_RES_ORI_DIR = "dataset/low_res_ori"    # 条件1：低分辨率原图（256×256）
SEG_MASK_DIR = "dataset/seg_mask"          # 条件2：语义分割图（256×256，与低分辨率图结构匹配）
HIGH_RES_ORI_DIR = "dataset/high_res_ori"  # 目标：高分辨率原图（512×512）
# 训练参数（显存优先，12G卡重点调BATCH_SIZE）
BATCH_SIZE = 2  # 12G=1/2，24G=4/8，32G=8/16，建议偶数
EPOCHS = 50     # 数据集≤1000张设50，≥10000张设20，按需调整
LEARNING_RATE = 1e-4  # LoRA训练用1e-4，全量训练改1e-5
WARMUP_STEPS = 100    # 学习率预热步数，固定即可
SAVE_STEPS = 500      # 每500步保存一次模型
LOG_STEPS = 100       # 每100步记录训练日志
# 图像尺寸（与数据集严格一致，不可随意修改）
CONDITION_SIZE = (256, 256)  # 双条件图统一尺寸
TARGET_SIZE = (512, 512)     # 目标高分辨率图尺寸
# LoRA轻量化训练配置（显存不足必开，12G卡默认值即可）
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# 输出路径（自动创建，无需手动建）
OUTPUT_DIR = "controlnet_seg_guided_sr_model"  # 模型保存目录
LOG_DIR = "tensorboard_logs_seg_guided"        # 训练日志目录
# ===========================================================================

# 自动创建输出/日志目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

# 1. 定义【低分辨率图+分割图】三线配对数据集类
class SegGuidedSRDataset(Dataset):
    def __init__(self, low_res_ori_dir, seg_mask_dir, high_res_ori_dir, transform_cond=None, transform_tgt=None):
        # 按文件名排序，确保三线严格配对（核心！避免错位）
        self.img_names = sorted([f for f in os.listdir(low_res_ori_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        # 构建三线配对文件路径
        self.low_res_ori_paths = [os.path.join(low_res_ori_dir, f) for f in self.img_names]
        self.seg_mask_paths = [os.path.join(seg_mask_dir, f) for f in self.img_names]
        self.high_res_ori_paths = [os.path.join(high_res_ori_dir, f) for f in self.img_names]
        # 强制验证配对数量一致，避免训练报错
        assert len(self.low_res_ori_paths) == len(self.seg_mask_paths) == len(self.high_res_ori_paths), \
            "低分辨率原图、语义分割图、高分辨率原图数量必须一致！"
        self.transform_cond = transform_cond  # 双条件图统一变换（尺寸/归一化）
        self.transform_tgt = transform_tgt    # 目标图变换

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 加载双条件图和目标图，均转为RGB格式（适配ControlNet输入）
        # 加载低分辨率原图
        low_res_ori = cv2.imread(self.low_res_ori_paths[idx])
        low_res_ori = cv2.cvtColor(low_res_ori, cv2.COLOR_BGR2RGB)
        # 加载语义分割图（无论原图是单通道/多通道，均转RGB保证3通道）
        seg_mask = cv2.imread(self.seg_mask_paths[idx])
        if len(seg_mask.shape) == 2:  # 单通道分割图转RGB
            seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
        else:
            seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2RGB)
        # 加载目标高分辨率原图
        high_res_ori = cv2.imread(self.high_res_ori_paths[idx])
        high_res_ori = cv2.cvtColor(high_res_ori, cv2.COLOR_BGR2RGB)

        # 双条件图统一做尺寸/归一化变换
        if self.transform_cond:
            low_res_ori = self.transform_cond(image=low_res_ori)["image"]
            seg_mask = self.transform_cond(image=seg_mask)["image"]
        # 目标图单独做变换
        if self.transform_tgt:
            high_res_ori = self.transform_tgt(image=high_res_ori)["image"]

        # 【核心】双条件图通道拼接：3(低分辨率图)+3(分割图)=6通道，适配ControlNet单输入
        combined_cond = torch.cat([low_res_ori, seg_mask], dim=0)  # 输出：6×256×256

        # 生成空文本嵌入（超分任务无需文本提示，SD1.5固定77×768维度）
        prompt_emb = torch.zeros(77, 768)

        return {
            "combined_cond": combined_cond,  # 6通道双条件拼接图（ControlNet核心输入）
            "target_img": high_res_ori,      # 3通道高分辨率目标图
            "prompt_emb": prompt_emb         # 空文本嵌入
        }

# 2. 定义数据变换（归一化到[-1,1]，SD/ControlNet标准输入格式）
def get_transforms():
    # 双条件图变换：固定尺寸+归一化+转Tensor
    transform_cond = A.Compose([
        A.Resize(height=CONDITION_SIZE[0], width=CONDITION_SIZE[1]),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化到[-1,1]
        ToTensorV2()
    ])
    # 目标图变换：与双条件图逻辑一致，仅尺寸不同
    transform_tgt = A.Compose([
        A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1]),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    return transform_cond, transform_tgt

# 3. 加载数据集和数据加载器（适配Windows/Linux多进程）
transform_cond, transform_tgt = get_transforms()
dataset = SegGuidedSRDataset(LOW_RES_ORI_DIR, SEG_MASK_DIR, HIGH_RES_ORI_DIR, transform_cond, transform_tgt)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 打乱数据，提升泛化能力
    num_workers=4 if os.name != "nt" else 0,  # Windows设0，避免多进程报错
    pin_memory=True,  # 锁页内存，提升训练速度
    drop_last=True    # 丢弃不完整批次，避免维度错误
)

# 4. 初始化ControlNet（核心修改：输入通道改为6）和SD1.5 UNet
print("开始加载SD1.5和ControlNet基础模型（首次运行自动下载，约4G）...")
# 加载ControlNet基础模型（SD1.5专用）
controlnet = ControlNetModel.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="controlnet",
    torch_dtype=torch.float16,  # 半精度训练，减少显存占用
    variant="fp16"
).to(DEVICE)
# 【关键修改】将ControlNet输入卷积层从3通道改为6通道，适配双条件拼接图
controlnet.conv_in = nn.Conv2d(
    in_channels=6,  # 3+3通道，核心适配点
    out_channels=controlnet.conv_in.out_channels,
    kernel_size=controlnet.conv_in.kernel_size,
    stride=controlnet.conv_in.stride,
    padding=controlnet.conv_in.padding
).to(DEVICE, dtype=torch.float16)

# 加载SD1.5的UNet（固定不训练，仅用于噪声预测）
unet = UNet2DConditionModel.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="unet",
    torch_dtype=torch.float16,
    variant="fp16"
).to(DEVICE)
# 加载SD1.5标准噪声调度器
noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_PATH, subfolder="scheduler")

# 5. LoRA轻量化训练配置（分割引导任务无修改，直接复用）
if USE_LORA:
    print("启用LoRA轻量化训练，打印可训练参数占比...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # ControlNet核心注意力层
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CONTROLNET"  # 明确任务类型为ControlNet
    )
    controlnet = get_peft_model(controlnet, lora_config)
    controlnet.print_trainable_parameters()  # 打印可训练参数（通常<1%，显存友好）

# 6. 优化器和学习率调度器（SD/ControlNet训练标准配置：余弦退火+预热）
optimizer = AdamW(
    filter(lambda p: p.requires_grad, controlnet.parameters()),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2
)
total_steps = len(dataloader) * EPOCHS
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

# 7. 核心训练步骤（双条件输入，与单条件逻辑一致，仅输入为6通道）
def train_step(batch):
    controlnet.train()
    unet.eval()  # 固定UNet，仅训练ControlNet（ControlNet训练核心逻辑）
    batch_size = batch["target_img"].shape[0]

    # 加载数据并移至GPU，半精度训练
    combined_cond = batch["combined_cond"].to(DEVICE, dtype=torch.float16)  # 6×256×256
    target_img = batch["target_img"].to(DEVICE, dtype=torch.float16)        # 3×512×512
    prompt_emb = batch["prompt_emb"].to(DEVICE, dtype=torch.float16)        # 77×768

    # 生成与目标图同尺寸的随机噪声（SD扩散训练核心）
    noise = torch.randn_like(target_img, dtype=torch.float16).to(DEVICE)
    # 随机选择噪声步数
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=DEVICE).long()
    # 对目标图添加噪声，得到带噪图
    noisy_target = noise_scheduler.add_noise(target_img, noise, timesteps)

    # ControlNet基于6通道双条件图生成控制特征
    controlnet_hidden_states = controlnet(
        combined_cond,
        timesteps,
        encoder_hidden_states=prompt_emb,
        return_dict=False
    )[0]

    # UNet基于控制特征预测噪声（禁用梯度，不训练UNet）
    with torch.no_grad():
        unet_output = unet(
            noisy_target,
            timesteps,
            encoder_hidden_states=prompt_emb,
            down_block_additional_residuals=[controlnet_hidden_states],
            mid_block_additional_residual=controlnet_hidden_states,
            return_dict=False
        )[0]

    # 计算MSE损失：预测噪声与真实噪声的差值（SD/ControlNet标准损失函数）
    loss = nn.functional.mse_loss(unet_output, noise, reduction="mean")

    # 反向传播与参数优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return loss.item()

# 8. 训练主循环（含断点续训、模型保存、日志记录）
print(f"开始训练分割引导的ControlNet超分模型，总训练步数：{total_steps}")
global_step = 0
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    # 进度条展示训练过程
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=120)
    for batch in pbar:
        loss = train_step(batch)
        epoch_loss += loss
        global_step += 1

        # 记录训练日志（损失、学习率），可通过TensorBoard查看
        if global_step % LOG_STEPS == 0:
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

        # 分步保存模型（支持断点续训）
        if global_step % SAVE_STEPS == 0 or global_step == total_steps:
            # 保存LoRA权重（若启用）
            if USE_LORA:
                lora_save_path = os.path.join(OUTPUT_DIR, f"lora_step_{global_step}")
                controlnet.save_pretrained(lora_save_path)
            # 保存全量ControlNet权重（.pth格式）
            full_save_path = os.path.join(OUTPUT_DIR, f"controlnet_step_{global_step}.pth")
            torch.save(controlnet.state_dict(), full_save_path)
            pbar.write(f"模型已保存至：{full_save_path}")

        # 进度条实时展示步损失和平均损失
        pbar.set_postfix({"step_loss": f"{loss:.4f}", "avg_loss": f"{epoch_loss/global_step:.4f}"})

    # 每个Epoch结束后保存一次模型，方便按轮数选择
    epoch_save_path = os.path.join(OUTPUT_DIR, f"controlnet_epoch_{epoch+1}.pth")
    torch.save(controlnet.state_dict(), epoch_save_path)
    print(f"第{epoch+1}轮模型保存完成：{epoch_save_path}")

# 9. 导出【Stable Diffusion WebUI兼容】的最终模型（关键！可直接加载）
print("训练完成，开始导出WebUI兼容模型...")
final_model_name = "controlnet_seg_guided_sr_webui.pth"
final_model_path = os.path.join(OUTPUT_DIR, final_model_name)
# 按WebUI ControlNet标准格式保存，包含配置信息，避免加载失败
torch.save(
    {
        "state_dict": controlnet.state_dict(),
        "config": controlnet.config,
        "model_type": "controlnet",
        "base_model": "sd15",
        "input_channels": 6,  # 标记6通道输入，方便WebUI识别
        "input_size": CONDITION_SIZE,
        "target_size": TARGET_SIZE
    },
    final_model_path,
    _use_new_zipfile_serialization=False
)
print(f"WebUI兼容模型导出完成：{final_model_path}")
writer.close()
print("所有训练流程结束！")