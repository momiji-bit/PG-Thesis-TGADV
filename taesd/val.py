import os
import cv2
import numpy as np
import torch
from taesd import TAESD  # TAESD 模型 (Model)
from PIL import Image
import torchvision.transforms.functional as TF  # TF: torchvision.transforms.functional
from tqdm import tqdm
import logging

# 设置日志配置 (Logging configuration)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 设置设备 (Device setup)：优先使用 GPU，其次 mps，最后 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logging.info(f"使用设备 (Using device): {device}")

# 初始化 TAESD 模型并移动到设备上 (Initialize and move model to device)
taesd = TAESD().to(device)


def process_video(input_video_path, output_video_path, batch_size=16):
    """
    处理单个视频：对视频的每一帧进行 TAESD 的编码、量化、反量化与解码，
    并将生成的帧保存到输出视频中。通过批处理（Batch Processing）来加速。

    参数 (Parameters)：
    - input_video_path: 输入视频路径 (input video path)
    - output_video_path: 输出视频路径 (output video path)
    - batch_size: 批处理大小 (batch size)，默认16
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"无法打开视频: {input_video_path}")
        return

    # 获取视频参数 (Get video parameters)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式 (codec)

    # 确保输出目录存在 (Ensure output directory exists)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"处理视频: {input_video_path}，帧数: {total_frames}")

    pbar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    frame_buffer = []  # 用于存储待处理帧的列表 (Buffer for storing frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将 BGR 格式的帧转换为 RGB，并转换为 PIL Image (Convert BGR to RGB and then to PIL Image)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(frame_rgb)
        # 转换为 tensor，范围 [0,1] (Convert to tensor in range [0,1])
        im_tensor = TF.to_tensor(pil_im)  # shape: [C, H, W]
        frame_buffer.append(im_tensor)

        # 当达到批处理大小或视频读完时，进行批量处理
        if len(frame_buffer) == batch_size:
            process_batch(frame_buffer, width, height, out_writer, pbar)
            frame_buffer = []

    # 处理剩余不足一个批次的帧
    if len(frame_buffer) > 0:
        process_batch(frame_buffer, width, height, out_writer, pbar)

    pbar.close()
    cap.release()
    out_writer.release()
    logging.info(f"已生成视频: {output_video_path}")


def process_batch(frame_list, width, height, out_writer, pbar):
    """
    对一个批次（Batch）的帧进行处理：
    - 堆叠成 batch tensor (stack into batch tensor)
    - 编码 (encode)
    - 缩放潜变量 (scale latents)
    - 量化 (quantize)
    - 模拟保存与加载（通过 PIL 转换） (simulate saving and loading via PIL conversion)
    - 反量化 (unscale latents)
    - 解码 (decode)
    - 将每帧写入输出视频 (write each frame to video)
    """
    # 将列表堆叠为 batch tensor，shape: [B, C, H, W]
    batch_tensor = torch.stack(frame_list).to(device)

    # 编码 (Encoding)
    latent = taesd.encoder(batch_tensor)
    # 潜变量缩放 (Scaling latents)
    scaled_latent = taesd.scale_latents(latent)
    # 量化 (Quantization)：乘以255并取整，再转换为字节类型 (multiply by 255, round, and convert to byte)
    quantized_latent = scaled_latent.mul(255).round().byte()

    # 模拟保存与加载的过程 (Simulate encoded saving & loading)
    reloaded_list = []
    for i in range(quantized_latent.shape[0]):
        # 将量化结果转换为 PIL Image (Convert quantized latent to PIL Image)
        encoded_pil = TF.to_pil_image(quantized_latent[i])
        # 再转换回 tensor (Convert back to tensor)
        reloaded_tensor = TF.to_tensor(encoded_pil).to(device)
        reloaded_list.append(reloaded_tensor)
    # 堆叠成 batch tensor，shape: [B, C, H, W]
    reloaded_tensor = torch.stack(reloaded_list)

    # 反量化 (Unscaling latents)
    unscaled_latent = taesd.unscale_latents(reloaded_tensor)
    # 解码 (Decoding)，结果范围限制在 [0,1] (clamp to [0,1])
    decoded_tensor = taesd.decoder(unscaled_latent).clamp(0, 1)

    # 遍历 batch 内每一帧进行写入 (Convert each decoded tensor to frame and write to video)
    for i in range(decoded_tensor.shape[0]):
        decoded_pil = TF.to_pil_image(decoded_tensor[i])
        # PIL Image 转 numpy 数组，并从 RGB 转换为 BGR (Convert PIL Image to numpy array and then to BGR)
        decoded_frame = cv2.cvtColor(np.array(decoded_pil), cv2.COLOR_RGB2BGR)
        # 若解码后图像尺寸与原视频不一致，则进行调整 (Resize if dimensions do not match)
        if decoded_frame.shape[1] != width or decoded_frame.shape[0] != height:
            decoded_frame = cv2.resize(decoded_frame, (width, height))
        out_writer.write(decoded_frame)
        pbar.update(1)


def main(input_root, output_root, batch_size=16):
    """
    遍历输入视频根目录，按照目录层级读取视频，
    并对每个视频进行 TAESD 生成（VAE 生成），
    生成结果按相同目录层级输出。

    参数 (Parameters)：
    - input_root: 输入视频根目录 (input root directory)
    - output_root: 输出视频根目录 (output root directory)
    - batch_size: 批处理大小 (batch size)
    """
    # 获取所有子目录（类别） (Get subdirectories/categories)
    categories = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if not categories:
        logging.warning("未发现子目录，可能根目录下直接为视频文件。")
        categories = [""]

    for cat in categories:
        input_cat_dir = os.path.join(input_root, cat)
        output_cat_dir = os.path.join(output_root, cat)
        os.makedirs(output_cat_dir, exist_ok=True)

        # 遍历当前目录下所有视频文件 (Iterate over video files in the current directory)
        video_files = [f for f in os.listdir(input_cat_dir) if os.path.isfile(os.path.join(input_cat_dir, f))]
        for video in video_files:
            input_video_path = os.path.join(input_cat_dir, video)
            # 输出视频文件名添加后缀 "_decoded" 表示 TAESD 生成结果
            video_name, ext = os.path.splitext(video)
            output_video_path = os.path.join(output_cat_dir, f"{video_name}{ext}")

            logging.info(f"开始处理视频: {input_video_path}")
            process_video(input_video_path, output_video_path, batch_size=batch_size)


if __name__ == "__main__":
    # 设置输入视频根目录和输出视频根目录 (Set input and output root directories)
    input_root = "UCF-101"  # 例如："input_videos/类别1/video.mp4"
    output_root = "output_videos"  # 例如："output_videos/类别1/video_decoded.mp4"
    # 调整批处理大小 (Adjust batch size to improve speed)
    main(input_root, output_root, batch_size=16)
