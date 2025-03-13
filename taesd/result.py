import json
import numpy as np

# 读取JSON文件 (Read JSON file)
file_path = "checkpoint_.json"
with open(file_path, "r") as file:
    data = json.load(file)

# 存储每个类别的统计信息 (Store statistics for each category)
category_metrics = {}

# 遍历所有类别 (Iterate over all categories)
for category, videos in data.items():
    psnr_values = []
    ssim_values = []
    lpips_values = []

    # 遍历该类别下所有视频 (Iterate over videos in the category)
    for video, metrics in videos.items():
        if metrics.get("psnr") is not None:
            psnr_values.append(metrics["psnr"])
        if metrics.get("ssim") is not None:
            ssim_values.append(metrics["ssim"])
        if metrics.get("lpips") is not None:
            lpips_values.append(metrics["lpips"])

    # 计算平均值 (Compute average values); 若列表为空，则设为0 (if list is empty, set to 0)
    category_metrics[category] = {
        "psnr_mean": np.mean(psnr_values) if psnr_values else 0,
        "ssim_mean": np.mean(ssim_values) if ssim_values else 0,
        "lpips_mean": np.mean(lpips_values) if lpips_values else 0,
        "video_count": len(psnr_values),  # 仅计算有PSNR值的视频数量 (Only count videos with PSNR value)
    }

# 计算所有类别的加权平均值 (Compute weighted averages for all categories)
total_videos = sum(cat["video_count"] for cat in category_metrics.values() if cat["video_count"] > 0)

if total_videos > 0:
    weighted_psnr = sum(cat["psnr_mean"] * cat["video_count"] for cat in category_metrics.values()) / total_videos
    weighted_ssim = sum(cat["ssim_mean"] * cat["video_count"] for cat in category_metrics.values()) / total_videos
    weighted_lpips = sum(cat["lpips_mean"] * cat["video_count"] for cat in category_metrics.values()) / total_videos
else:
    weighted_psnr, weighted_ssim, weighted_lpips = 0, 0, 0

# 输出CSV格式数据，便于直接复制到Excel中 (Output CSV-formatted data for Excel)
print("| Classes | PSNR↑ | SSIM↑ | LPIPS↓ | count |")
print('| -------------- | ----------------- | ------------------ | ------------------- | ---- |')
for category, metrics in category_metrics.items():
    print(f"| {category} | {metrics['psnr_mean']} | {metrics['ssim_mean']} | {metrics['lpips_mean']} | {metrics['video_count']} |")

print("\n加权指标(Weighted Metrics)")
print("指标(Metric),数值(Value)")
print(f"加权PSNR, {weighted_psnr}")
print(f"加权SSIM, {weighted_ssim}")
print(f"加权LPIPS, {weighted_lpips}")
