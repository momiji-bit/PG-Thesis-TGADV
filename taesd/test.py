import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim  # SSIM: Structural Similarity Index Measure（结构相似度指数）
import warnings
from tqdm import tqdm
import json
import logging

# 过滤掉 UserWarning 警告（例如 'pretrained' 警告）
warnings.filterwarnings("ignore", category=UserWarning)

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metrics.log", mode='a', encoding='utf-8'),  # 保存日志到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# 设置设备，优先使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 LPIPS 模型，使用 AlexNet backbone
# LPIPS: Learned Perceptual Image Patch Similarity（学习感知图像块相似度）
lpips_model = lpips.LPIPS(net='alex').to(device)


def compute_video_metrics(gt_video_path, vae_video_path, batch_size=16):
    """
    计算单个视频的指标（PSNR, SSIM, LPIPS）。
    GT: Ground Truth（真实视频）
    VAE: VAE 生成的视频

    术语：
    - PSNR: Peak Signal-to-Noise Ratio（峰值信噪比）
    - SSIM: Structural Similarity Index Measure（结构相似度指数）
    - LPIPS: Learned Perceptual Image Patch Similarity（学习感知图像块相似度）
    """
    cap_gt = cv2.VideoCapture(gt_video_path)
    cap_vae = cv2.VideoCapture(vae_video_path)

    psnr_list = []
    ssim_list = []
    lpips_list = []
    lpips_frames_gt = []
    lpips_frames_vae = []

    while True:
        ret_gt, frame_gt = cap_gt.read()
        ret_vae, frame_vae = cap_vae.read()
        if not ret_gt or not ret_vae:
            break

        if frame_gt.shape != frame_vae.shape:
            logging.warning(f"帧尺寸不匹配: {gt_video_path} 与 {vae_video_path}")
            continue

        # 计算 PSNR（峰值信噪比）
        psnr_val = cv2.PSNR(frame_gt, frame_vae)
        psnr_list.append(psnr_val)

        # 计算 SSIM（结构相似度指数）
        ssim_val = ssim(
            frame_gt,
            frame_vae,
            win_size=3,  # 小窗口保证适用于小尺寸图像
            channel_axis=2,  # 指定颜色通道所在轴（HWC格式中的 C 轴）
            data_range=frame_gt.max() - frame_gt.min()
        )
        ssim_list.append(ssim_val)

        # 为 LPIPS 计算准备：转换 BGR 到 RGB
        frame_gt_rgb = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB)
        frame_vae_rgb = cv2.cvtColor(frame_vae, cv2.COLOR_BGR2RGB)
        lpips_frames_gt.append(frame_gt_rgb)
        lpips_frames_vae.append(frame_vae_rgb)

        # 达到批处理大小后，一次性计算 LPIPS
        if len(lpips_frames_gt) == batch_size:
            batch_gt = np.stack(lpips_frames_gt, axis=0)
            batch_vae = np.stack(lpips_frames_vae, axis=0)
            batch_gt_tensor = torch.from_numpy(batch_gt).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1.0
            batch_vae_tensor = torch.from_numpy(batch_vae).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1.0
            with torch.no_grad():
                lpips_batch = lpips_model(batch_gt_tensor, batch_vae_tensor)
            lpips_list.extend(lpips_batch.view(-1).cpu().numpy().tolist())
            lpips_frames_gt.clear()
            lpips_frames_vae.clear()

    # 处理剩余不足一个批次的帧
    if len(lpips_frames_gt) > 0:
        batch_gt = np.stack(lpips_frames_gt, axis=0)
        batch_vae = np.stack(lpips_frames_vae, axis=0)
        batch_gt_tensor = torch.from_numpy(batch_gt).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1.0
        batch_vae_tensor = torch.from_numpy(batch_vae).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1.0
        with torch.no_grad():
            lpips_batch = lpips_model(batch_gt_tensor, batch_vae_tensor)
        lpips_list.extend(lpips_batch.view(-1).cpu().numpy().tolist())

    cap_gt.release()
    cap_vae.release()

    video_metrics = {
        'psnr': np.mean(psnr_list) if psnr_list else None,
        'ssim': np.mean(ssim_list) if ssim_list else None,
        'lpips': np.mean(lpips_list) if lpips_list else None
    }
    return video_metrics


def load_checkpoint(checkpoint_path):
    """
    加载检查点文件（Checkpoint File）。
    """
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        logging.info(f"加载检查点: {checkpoint_path}")
    else:
        checkpoint = {}
        logging.info("未找到检查点，初始化新检查点。")
    return checkpoint


def save_checkpoint(checkpoint_path, data):
    """
    保存检查点到文件（Save Checkpoint）。
    """
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    # logging.info(f"保存检查点: {checkpoint_path}")


def main(gt_root, vae_root, batch_size=16):
    """
    遍历所有类别（子目录），计算每个类别及总体的指标，并实现中断恢复（Resume on Interrupt）。

    gt_root: GT 视频根目录（Ground Truth）
    vae_root: VAE 生成视频根目录
    """
    checkpoint_path = "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)

    overall_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    category_metrics = {}

    # 获取所有子目录名称（类别名称）
    categories = [d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))]

    try:
        for cat in categories:
            logging.info(f"处理类别: {cat}")
            gt_cat_path = os.path.join(gt_root, cat)
            vae_cat_path = os.path.join(vae_root, cat)
            video_files = [f for f in os.listdir(gt_cat_path) if os.path.isfile(os.path.join(gt_cat_path, f))]
            cat_psnr = []
            cat_ssim = []
            cat_lpips = []

            # 检查该类别是否已有处理记录
            if cat in checkpoint:
                processed_videos = set(checkpoint[cat].keys())
                # logging.info(f"类别 {cat} 已处理视频: {processed_videos}")
            else:
                checkpoint[cat] = {}
                processed_videos = set()

            for video in tqdm(video_files, desc=f"Processing {cat}"):
                if video in processed_videos:
                    # logging.info(f"跳过已处理视频: {video}")
                    # 从检查点加载已处理视频的指标
                    metrics = checkpoint[cat][video]
                    if metrics['psnr'] is not None:
                        overall_metrics['psnr'].append(metrics['psnr'])
                        cat_psnr.append(metrics['psnr'])
                    if metrics['ssim'] is not None:
                        overall_metrics['ssim'].append(metrics['ssim'])
                        cat_ssim.append(metrics['ssim'])
                    if metrics['lpips'] is not None:
                        overall_metrics['lpips'].append(metrics['lpips'])
                        cat_lpips.append(metrics['lpips'])
                    continue

                gt_video_path = os.path.join(gt_cat_path, video)
                vae_video_path = os.path.join(vae_cat_path, video[:-4]+'_decoded'+video[-4:])

                if not os.path.exists(vae_video_path):
                    logging.warning(f"在 {vae_cat_path} 下找不到对应视频: {video}")
                    continue

                metrics = compute_video_metrics(gt_video_path, vae_video_path, batch_size=batch_size)
                checkpoint[cat][video] = metrics
                save_checkpoint(checkpoint_path, checkpoint)

                if metrics['psnr'] is not None:
                    overall_metrics['psnr'].append(metrics['psnr'])
                    cat_psnr.append(metrics['psnr'])
                if metrics['ssim'] is not None:
                    overall_metrics['ssim'].append(metrics['ssim'])
                    cat_ssim.append(metrics['ssim'])
                if metrics['lpips'] is not None:
                    overall_metrics['lpips'].append(metrics['lpips'])
                    cat_lpips.append(metrics['lpips'])
                # logging.info(f"视频 {video} 的指标: {metrics}")

            if cat_psnr:
                category_metrics[cat] = {
                    'psnr': np.mean(cat_psnr),
                    'ssim': np.mean(cat_ssim),
                    'lpips': np.mean(cat_lpips),
                    'video_count': len(cat_psnr)
                }
            logging.info(f"类别 {cat} 的指标: {category_metrics.get(cat, {})}")

        overall_avg = {
            'psnr': np.mean(overall_metrics['psnr']) if overall_metrics['psnr'] else None,
            'ssim': np.mean(overall_metrics['ssim']) if overall_metrics['ssim'] else None,
            'lpips': np.mean(overall_metrics['lpips']) if overall_metrics['lpips'] else None,
            'video_count': len(overall_metrics['psnr'])
        }

        logging.info("总指标 (Overall Metrics):")
        logging.info(overall_avg)
        logging.info("\n各类别指标 (Category Metrics):")
        for cat, metrics in category_metrics.items():
            logging.info(f"{cat}: {metrics}")
        print("处理完成。检查日志文件 metrics.log 以获取详细信息。")

    except KeyboardInterrupt:
        logging.info("检测到中断信号，程序退出。")
        print("程序被中断，当前进度已保存到检查点。")


if __name__ == "__main__":
    # 请替换为实际的 GT 视频根目录（Ground Truth）和 VAE 生成视频根目录
    gt_root = r"UCF-101"
    vae_root = "output_videos"
    # 可根据需要调整 batch_size（批处理大小）
    main(gt_root, vae_root, batch_size=32)
