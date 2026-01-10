from ConfigLoader import *
import os
import shutil
import torch
import numpy as np
from skimage import transform as trans
from torchvision.transforms import v2

def split_filename(filename):
    basename = os.path.basename(filename)
    file_idx_name, file_extension = os.path.splitext(basename)
    return file_idx_name, file_extension

def img_2_tensor(img=None):
    if img is not None:
        # 将 OpenCV 图像（BGR 格式）转换为 RGB 格式
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 将图像转换为 PyTorch 张量
        img = torch.from_numpy(img.astype('uint8')).permute(2, 0, 1).to('cuda')
        return img
    else:
        return None

def affine_transform(img, kpss, output_size, scale_factor=0.75, center_offset_y=-20):
    """
        对图像进行仿射变换，使其符合目标大小和关键点对齐要求

        参数:
            img: 输入图像
            kpss: 输入图像的人脸关键点
            output_size: 目标输出图像大小 (int) - 正方形
            scale_factor: 控制裁剪区域的缩放比例，默认为1.0（无缩放）
            center_offset_y: 垂直方向的中心偏移量（像素），正数向下偏移，负数向上偏移

        返回:
            transformed_img: 仿射变换后的图像
            aligned_kpss: 变换后的人脸关键点
    """
    # 定义标准的 ArcFace 关键点（用于 112x112 大小）
    arcface_dst_112 = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32
    )

    # 计算目标大小与原 ArcFace 大小的比例
    base_scale_factor = output_size / 112.0

    # 调整目标关键点到新的输出大小
    dst = arcface_dst_112 * base_scale_factor * scale_factor
    #dst[:, 0] += 8.0 * scale_factor  # 同样应用偏移量

    # 移动关键点，使裁剪区域以原关键点为中心
    center_offset = (dst - arcface_dst_112 * base_scale_factor).mean(axis=0)
    dst -= center_offset

    # 应用垂直方向的中心偏移
    dst[:, 1] += center_offset_y

    # 计算相似变换
    tform = trans.SimilarityTransform()
    tform.estimate(kpss, dst)

    # 应用仿射变换
    transformed_img = v2.functional.affine(
        img,
        tform.rotation * 57.2958,
        (tform.translation[0], tform.translation[1]+center_offset_y),
        tform.scale,0,
        center=(0, 0))

    # 裁剪图像到目标大小
    transformed_img = v2.functional.crop(transformed_img, 0, 0, output_size, output_size)

    # 计算变换后的人脸关键点
    #aligned_kpss = self.transform_kpss(kpss, tform, output_size)

    return transformed_img

def prepare_workspace():
    work_path = G_CONFIG["work_path"]
    out=[]
    for filename in os.listdir(work_path):
        file_idx_name, _ = os.path.splitext(filename)
        file_path = os.path.join(work_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_out_path = os.path.join(work_path, file_idx_name)
            if os.path.exists(video_out_path):
                shutil.rmtree(video_out_path)
            os.makedirs(video_out_path, exist_ok=True)
            face_out_path = os.path.join(video_out_path, "face_set")
            if os.path.exists(face_out_path):
                shutil.rmtree(face_out_path)
            os.makedirs(face_out_path, exist_ok=True)
            out.append((file_idx_name, filename, file_path, video_out_path, face_out_path))
    return out

def prepare_face_set(face_in_path):
    img_list = []
    for filename in os.listdir(face_in_path):
        file_path = os.path.join(face_in_path, filename)
        # 检查文件扩展名，只处理图片文件
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.png')):
            file_idx_name, file_extension = os.path.splitext(filename)
            img_list.append((file_idx_name, file_path))
    return img_list
