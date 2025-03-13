import os
import torch
import cv2  # OpenCV，用于视频读取和写入 (OpenCV for video I/O)
import gc
from taesdv import TAESDV
from tqdm import tqdm  # 引入 tqdm 用于进度条显示 (Progress bar)


# 定义视频读取类 (VideoTensorReader)
class VideoTensorReader:
    def __init__(self, video_file_path):
        self.cap = cv2.VideoCapture(video_file_path)
        assert self.cap.isOpened(), f"无法加载视频文件 {video_file_path}"
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration  # 视频结束或出错 (End of video or error)
        # 将 BGR 转换为 RGB，并转换为 Tensor (Convert BGR to RGB and then to tensor; shape: [C, H, W])
        return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)


# 定义视频写入类 (VideoTensorWriter)
class VideoTensorWriter:
    def __init__(self, video_file_path, width_height, fps=30):
        self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width_height)
        # self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'HFYU'), fps, width_height)
        assert self.writer.isOpened(), f"无法创建视频写入器 {video_file_path}"

    def write(self, frame_tensor):
        # 确保 frame_tensor 是 3D 且第一个维度为 3 (RGB channels)
        assert frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3, f"{frame_tensor.shape}??"
        # 将 tensor 转换为 numpy 数组，并从 RGB 转换回 BGR 写入文件 (Convert tensor from RGB to BGR and write)
        self.writer.write(cv2.cvtColor(frame_tensor.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))

    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.release()


def main():
    # 设置设备 (Set device) 与数据类型 (and data type)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16

    # 加载 TAESDV 模型，并设置为评估模式 (Load TAESDV model and set to evaluation mode)
    taesdv = TAESDV().to(dev, dtype)
    taesdv.eval()

    # 输入与输出目录 (Input and output directories)
    input_dir = "UCF-101"  # 请将此处替换为你的输入目录 (replace with your input directory)
    output_dir = "output"  # 输出目录 (output directory)

    # 定义允许的视频文件扩展名 (Allowed video file extensions)
    video_extensions = {'.avi', '.mp4', '.mov', '.mkv'}

    # 预先遍历所有待处理视频文件 (Pre-traverse all video files to process)
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in video_extensions:
                video_files.append((root, file))

    # 使用 tqdm 进度条遍历视频文件 (Using tqdm progress bar to process videos)
    with tqdm(total=len(video_files), desc="处理视频 (Processing videos)") as pbar:
        for root, file in video_files:
            input_video_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_dir)
            out_dir_full = os.path.join(output_dir, rel_path)
            os.makedirs(out_dir_full, exist_ok=True)
            output_video_path = os.path.join(out_dir_full, file)

            status_msg = f"正在处理 {input_video_path}... "
            # 读取视频 (Read video)
            video_in = VideoTensorReader(input_video_path)
            frames = list(video_in)
            video = torch.stack(frames, 0)[None]  # [1, T, C, H, W]
            del frames

            vid_dev = video.to(dev, dtype).div_(255.0)

            with torch.no_grad():
                # if video.numel() < 100_000_000:
                #     # status_msg += "文件较小，采用并行处理 (Parallel processing) ... "
                #     vid_enc = taesdv.encode_video(vid_dev)
                #     # status_msg += "编码完成 (Encoded) ... "
                #     vid_dec = taesdv.decode_video(vid_enc)
                #     # status_msg += "解码完成 (Decoded) ... "
                # else:
                # status_msg += "文件较大，采用串行处理 (Serial processing) ... "
                vid_enc = taesdv.encode_video(vid_dev, parallel=False)
                # status_msg += "编码完成 (Encoded) ... "
                vid_dec = taesdv.decode_video(vid_enc, parallel=False)
                # status_msg += "解码完成 (Decoded) ... "

            writer = VideoTensorWriter(output_video_path, (vid_dec.shape[-1], vid_dec.shape[-2]),
                                       fps=int(round(video_in.fps)))
            frames_decoded = vid_dec.clamp(0, 1).mul(255).round().byte().cpu()[0]
            for frame in frames_decoded:
                writer.write(frame)
            # status_msg += f"输出保存至 {output_video_path}"

            # 更新进度条描述信息 (Update progress bar description)
            pbar.set_description(status_msg)

            # 清理内存 (Clean up memory)
            del video, vid_dev, vid_enc, vid_dec, video_in, frames_decoded
            torch.cuda.empty_cache()
            gc.collect()

            pbar.update(1)


if __name__ == "__main__":
    main()
