from ConfigLoader import logging
import Utils as utils
import subprocess

def video2img(video_path, out_path, exp_mod="I"):
    video_name, _ = utils.split_filename(video_path)
    # 定义ffmpeg命令
    if exp_mod == 'I':
        ffmpeg_frame = " -vf \"select=\'eq(pict_type\\,I)\'\" -vsync vfr"
    elif exp_mod == 'A':
        ffmpeg_frame = " -vf \"select=\'eq(pict_type\\,I)*mod(n\\,2)\'\" -vsync vfr"
    else:
        ffmpeg_frame = " -vf" + " \"fps=" + exp_mod + "\""
    ffmpeg_input = " -i \"" + video_path + "\""
    ffmpeg_output = " \"" + out_path + "/" + video_name + "_%04d.png\""
    ffmpeg_cmd = "./libs/ffmpeg.exe" + ffmpeg_input + ffmpeg_frame + ffmpeg_output

    try:
        logging.debug(f"Execute CMD: {ffmpeg_cmd}")
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            process = subprocess.Popen(ffmpeg_cmd)
        else:
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 等待进程结束
        process.wait()
    except Exception as e:
        logging.error(f"ffmpeg Error: {e}", exc_info=True)

def get_frame_info(video_path):
    try:
        # 执行 ffprobe 命令来获取视频帧信息
        command = [
            './libs/ffprobe.exe',
            '-loglevel', 'error',  # 设置日志级别为 error，减少不必要的日志输出
            '-select_streams', 'v:0',
            '-show_frames',
            '-show_entries', 'frame=pict_type',
            '-of', 'csv=p=0',
            video_path
        ]
        # 执行命令并获取输出
        output = subprocess.check_output(command).decode("utf-8")
        return output
    except Exception as e:
        logging.error(f"ffprobe Error: {e}", exc_info=True)
        return ""

def get_1m_frame_info(video_path):
    try:
        _, file_type  = utils.os.path.splitext(video_path)
        tmp_file_name = "tmp"+file_type
        # 使用 ffmpeg 提取前 1 分钟的视频
        command = [
            './libs/ffmpeg.exe',
            '-i', video_path,
            '-to', '60',  # 提取 60 秒（1 分钟）
            '-c', 'copy',  # 复制原始流，不重新编码
            '-y',  # 覆盖输出文件（如果存在）
            tmp_file_name
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 执行 ffprobe 命令来获取视频帧信息
        return get_frame_info(tmp_file_name)
    except Exception as e:
        logging.error(f"ffprobe Error: {e}", exc_info=True)
        return ""

if __name__ == '__main__':
    print(f"You are running {__file__}")
    logging.getLogger().setLevel(logging.DEBUG)

    print(f"Running test: video2img")
    video2img("./workspace/Demo1.mp4","./workspace/Demo1",'A')

    print(f"Running test: get_1m_frame_info")
    assert get_1m_frame_info("./workspace/Demo1.mp4"), "get_1m_frame_info Failed"
