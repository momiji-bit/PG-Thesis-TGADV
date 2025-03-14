import os
import cv2  # OpenCV库 (OpenCV library)
from collections import defaultdict
from tqdm import tqdm

def get_video_resolution(video_path):
    """
    获取视频分辨率 (Get video resolution)

    参数 (Parameters):
      video_path: 视频文件路径 (video file path)

    返回 (Returns):
      (width, height) 元组，如果无法读取则返回None (tuple, or None if cannot be read)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (width, height)

def main(directory):
    """
    遍历指定目录下所有二级目录中的avi视频，统计分辨率类型并记录文件路径，
    最后输出数量较少（出现次数最少）的分辨率及对应的视频文件路径

    Traverse all second-level directories under the specified directory for avi videos,
    count resolution types and record file paths, then output the resolution(s) with the least occurrences and the corresponding file paths.

    参数 (Parameters):
      directory: 指定的目录路径 (specified directory path)
    """
    resolution_stats = defaultdict(int)
    resolution_paths = defaultdict(list)

    # 遍历指定目录 (traverse the directory)
    for root, dirs, files in tqdm(os.walk(directory)):
        # 计算当前目录相对于指定目录的深度 (calculate depth relative to the specified directory)
        rel_path = os.path.relpath(root, directory)
        depth = rel_path.count(os.sep)
        # 仅处理二级目录 (only process second-level directories)
        if depth != 0:
            continue
        for file in files:
            if file.lower().endswith('.avi'):
                video_path = os.path.join(root, file)
                resolution = get_video_resolution(video_path)
                if resolution:
                    resolution_stats[resolution] += 1
                    resolution_paths[resolution].append(video_path)

    # 输出所有分辨率统计 (print all resolution statistics)
    print("视频分辨率统计 (Video Resolution Statistics):")
    for res, count in resolution_stats.items():
        print(f"分辨率 (Resolution): {res[0]}x{res[1]} - 数量 (Count): {count}")

    if not resolution_stats:
        print("没有找到视频文件 (No video files found).")
        return

    # 找到出现次数最少的分辨率 (find the resolution(s) with the minimum count)
    min_count = min(resolution_stats.values())
    rare_resolutions = [res for res, count in resolution_stats.items() if count == min_count]

    print("\n数量较少的分辨率及对应视频路径 (Rare resolutions and their video paths):")
    for res in rare_resolutions:
        print(f"分辨率 (Resolution): {res[0]}x{res[1]} - 数量 (Count): {min_count}")
        for path in resolution_paths[res]:
            print(f"  路径 (Path): {path}")

if __name__ == '__main__':
    # 用户输入指定目录 (User input for target directory)
    target_directory = r"C:\Users\gujih\iCloudDrive\UCL CGVI\TGADV\Datasets\UCF-101"
    main(target_directory)
