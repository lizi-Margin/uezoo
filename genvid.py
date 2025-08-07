import os
import re
import argparse
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def combine_3vids(video1_path, video2_path, video3_path, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    cap3 = cv2.VideoCapture(video3_path)
    
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*3, height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        
        if not (ret1 and ret2 and ret3):
            break
            
        combined = cv2.hconcat([frame1, frame2, frame3])
        out.write(combined)
    
    cap1.release()
    cap2.release()
    cap3.release()
    out.release()

def run_combine_3vids(target_dir):
    
    # 构建完整文件路径
    video1_path = os.path.join(target_dir, 'rgb.mp4')
    video2_path = os.path.join(target_dir, 'mask_filtered.mp4')
    video3_path = os.path.join(target_dir, 'REPLAY_rgb.mp4')
    output_path = os.path.join(target_dir, 'rgb|mask_filtered|REPLAY_rgb.mp4')
    
    # 调用函数合并视频
    combine_3vids(video1_path, video2_path, video3_path, output_path)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用cv2将n_xxx.png格式的图片序列转换为xxx.mp4视频')
    parser.add_argument('--input-dir')
    parser.add_argument('--fps', type=int, default=5)
    args = parser.parse_args()

    # 确定输出目录（与图片同目录）
    output_dir = args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    # 正则表达式匹配 n_xxx.png 格式的文件名
    pattern = re.compile(r'^(\d+)_(.+)\.png$')

    # 按序列名分组
    sequences = defaultdict(list)

    # 遍历目录中的所有png文件
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith('.png'):
            match = pattern.match(filename)
            if match:
                # 提取序号和序列名
                frame_num = int(match.group(1))
                seq_name = match.group(2)
                file_path = os.path.join(args.input_dir, filename)
                sequences[seq_name].append((frame_num, file_path))

    # 如果没有找到任何序列
    if not sequences:
        print("未找到符合格式的图片序列（格式应为：n_xxx.png）")
        return

    # 处理每个序列
    for seq_name, frames in tqdm(sequences.items(), desc="处理序列"):
        # 按帧序号排序
        frames.sort(key=lambda x: x[0])
        sorted_files = [f[1] for f in frames]

        # 输出视频文件名
        output_file = os.path.join(output_dir, f"{seq_name}.mp4")

        # 读取第一张图片获取尺寸
        first_frame = cv2.imread(sorted_files[0])
        if first_frame is None:
            tqdm.write(f"无法读取图片: {sorted_files[0]}，跳过该序列")
            continue
        
        height, width, channels = first_frame.shape

        # 定义视频编码器和创建VideoWriter对象
        # 使用MP4V编码器，生成mp4格式视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, args.fps, (width, height))

        if not out.isOpened():
            tqdm.write(f"无法创建视频文件: {output_file}，可能是编码器不支持")
            continue

        # 写入所有帧
        for file_path in tqdm(sorted_files, desc=f"处理 {seq_name}", leave=False):
            frame = cv2.imread(file_path)
            if frame is None:
                tqdm.write(f"警告：无法读取图片 {file_path}，已跳过")
                continue
            
            # 确保帧尺寸与视频一致
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)

        # 释放资源
        out.release()
        tqdm.write(f"成功生成视频: {output_file}")

    print("所有序列处理完成！")

    try:
        run_combine_3vids(output_dir)
    except Exception:
        print("Combine failed")




if __name__ == "__main__":
    main()
    