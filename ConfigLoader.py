import logging
logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s.%(msecs)03d|%(levelname)s|%(filename)s:%(lineno)d|%(funcName)s -> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
import json

"""
    说明：
    1. JSON配置文件不支持便捷写注释，且为避免普通用户误操作Python代码，核心配置注释集中在此处维护
    2. 本文件中G_CONFIG为默认配置，程序读取外部JSON配置文件后，会用读取到的参数覆盖此处默认值
    -----------------------------------------
    G_CONFIG作为全局配置参数使用，配置参数说明如下
    -----------------------------------------
    > work_path: 工作路径，程序会自动扫描并处理该路径下的所有视频文件（支持mp4/avi/mov/mkv格式）
    > exp_video_mode: 视频帧导出模式，仅支持以下有效值（非有效值会触发程序异常，暂未做异常保护）
        - "I": 仅导出视频的I帧（关键帧，画面完整度/清晰度最高）
        - 数字: 按帧率导出，如2=每秒导出2帧，支持分数，如1/2=每2秒导出1帧
        - "A": 自动模式（默认），结合exp_video_frame_limit参数使用，逻辑如下
            1) 先提取视频前1分钟的所有I帧
            2) 若I帧数量 ≤ exp_video_frame_limit → 按"I"模式导出全部I帧
            3) 若I帧数量 > exp_video_frame_limit → 按"I帧数量/2"导出（减半采样，避免I帧过多）
            注：自动模式是项目实践经验值，仅做1次减半处理，若仍需调整可手动切换为FPS数字模式
    > exp_video_frame_limit: 自动模式下I帧数量阈值，用于判断是否需要对I帧减半采样，取值建议20~50（默认30）
    > face_confidence_thresh: AI人脸检测置信度阈值（默认0.8），范围0~1
        - 过低（如<0.7）：易检出模糊/非人脸区域（无效人脸），增加后续计算量
        - 过高（如>0.9）：易漏检侧脸/遮挡人脸，降低检测召回率
        - 建议取值：0.75~0.85（平衡准确率和召回率）
    > img_max_face: 单张图片最大检出人脸数（默认10），超出该数量的人脸会被忽略
        - 取值建议：5~20（过少易漏检，过多增加单帧处理耗时）
        - 业务场景：适合单人/小群体视频场景，若为多人场景可调整至20~30
    > exp_face_scale_factor: 导出人脸区域的缩放系数（默认0.75），非必要不推荐修改
    > exp_face_z_fix: 导出人脸中心点的Z轴（垂直方向）偏移像素值（默认-20），非必要不推荐修改
    > exp_face_size: 导出人脸图片的分辨率（默认512），格式为N（代表N×N像素）
        - 取值建议：128/256/512/1024
    > export_cluster_min_len: 人脸聚类最小数量阈值（默认5）
        - 作用：聚类后，数量小于该值的人脸簇会被归类到"Others"类别，不单独导出
        - 取值建议：3~10（过滤少量零散的误检/低频次人脸，减少无效导出文件）
    > similarity_threshold: 人脸相似度阈值（默认0.4），范围0~1
        - 作用：判断两张人脸是否为同一人，用于聚类分组
        - 取值说明：
          - 过低: 不同人脸易被归为同一类，聚类准确率低
          - 过高: 同一人不同角度的人脸易被归为不同类，聚类召回率低
        - 建议取值: 无，根据视频中图像质量差别较大，聚类不满意请反复调整该值重试
"""
G_CONFIG={
  "work_path": "./workspace",
  "exp_video_mode": "A",
  "exp_video_frame_limit": 30,
  "face_confidence_thresh": 0.8,
  "img_max_face": 10,
  "exp_face_scale_factor": 0.75,
  "exp_face_z_fix": -20,
  "exp_face_size": 512,
  "export_cluster_min_len": 5,
  "similarity_threshold": 0.4
}

def load_configuration(file_path="VFC_Configuration.json"):
    # 打开并读取配置文件
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            global G_CONFIG
            G_CONFIG = json.load(file)
        logging.debug(f"Config Loaded: {G_CONFIG}")
    except Exception as e:
        logging.critical(f"load_configuration Error：{e}")
        return False
    return True
load_configuration()

if __name__ == '__main__':
    print(f"You are running {__file__}")
    logging.getLogger().setLevel(logging.DEBUG)

    print(f"Running test: load_configuration")
    assert load_configuration(), "load_configuration Failed"

    print(f"Done")