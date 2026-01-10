from ConfigLoader import logging
import Utils as utils
import FFmpegExport as fmp
from AIRuntime import Models
from Utils import torch
import time
import re
import pickle
import cv2
import numpy as np
import torch.nn.functional as F

class VFCWorks:
    def __init__(self):
        self.model = Models()
        self.config = utils.G_CONFIG
        self.extractor = None
        self.faces = dict()
        self.ffmpeg_time= 0.0
        self.kps_time = 0.0
        self.embed_time = 0.0
        self.similarity_time = 0.0
        self.export_time = 0.0
        self.cluster = {}
        self.similarity_matrix = None

    def clear_works(self):
        self.faces = dict()
        self.ffmpeg_time = 0.0
        self.kps_time = 0.0
        self.embed_time = 0.0
        self.similarity_time = 0.0
        self.export_time = 0.0
        self.cluster = {}
        self.similarity_matrix = None

    def export_data(self, data_name='data.pkl'):
        with open(data_name, 'wb') as f:
            pickle.dump(self.faces, f)

    def import_data(self, data_name='data.pkl'):
        try:
            with open(data_name, 'rb') as f:
                self.faces = pickle.load(f)
        except Exception as e:
            logging.critical(f"Error import: {e}", exc_info=True)

    def show_cost(self):
        print(f"VFCWorks Cost: FFMPEG:{self.ffmpeg_time:.3f}|KPS:{self.kps_time:.3f}|EMBED:{self.embed_time:.3f}|SIMILAR:{self.similarity_time:.3f}|EXPORT:{self.export_time:.3f}")

    def show_debug_values(self):
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            for k, v in self.cluster.items(): logging.debug(f"Cluster {k}: {len(v)}")
            logging.debug(f"Face Dict: {self.faces.keys()}")
            torch.set_printoptions(precision=2, sci_mode=False)
            logging.debug(f"Similarity Matrix: {self.similarity_matrix}")
        else:
            logging.info(f"Face Dict length: {len(self.faces)}")
            logging.info(f"Cluster Dict length: {len(self.cluster)}")

    def process_video_export(self, video_path, out_path):
        logging.info(f"Start...")
        try:
            start_time = time.time()
            if self.config["exp_video_mode"] == "A":
                frame_info=fmp.get_1m_frame_info(video_path)
                i_count = len(re.findall(r'I', frame_info))
                b_count = len(re.findall(r'B', frame_info))
                p_count = len(re.findall(r'P', frame_info))
                total_frames = len(frame_info.split('\n'))
                if (total_frames - i_count - b_count - p_count) <= 5:
                    logging.info(f"Auto mode -> Frames:{i_count}|{b_count}|{p_count}|{total_frames}")
                else:
                    logging.warning(f"Auto mode -> Frames not compare:{i_count}|{b_count}|{p_count}|{total_frames}")

                if i_count <= self.config["exp_video_frame_limit"]:
                    logging.info(f"use I export {i_count}|{self.config["exp_video_frame_limit"]}")
                    fmp.video2img(video_path, out_path, "I")
                else:
                    logging.info(f"use X export {i_count}|{self.config["exp_video_frame_limit"]}")
                    fmp.video2img(video_path, out_path, "A")
            else:
                fmp.video2img(video_path, out_path, self.config["exp_video_mode"])
            elapsed_time = time.time() - start_time
            self.ffmpeg_time += elapsed_time
            logging.debug(f"FFMPEG Cost time {elapsed_time:.3f}")
        except Exception as e:
            logging.critical(f"Error->process_video_export：{e}", exc_info=True)

    def process_one_img(self, img_name, file_path):
        logging.debug(f"{file_path} Start...")
        try:
            input_image = cv2.imread(file_path)
            if input_image is None:
                logging.error(f"Load failed Error:  {file_path}")
                return
        except Exception as e:
            logging.critical(f"OpenCV Error: {e}", exc_info=True)
            return

        # kps推理
        ts_img = utils.img_2_tensor(input_image)
        start_time = time.time()
        kpss_list = self.model.run_detect(ts_img, max_num=self.config["img_max_face"], score=self.config["face_confidence_thresh"])
        elapsed_time = time.time() - start_time
        logging.debug(f"kpss cost time {elapsed_time}")
        self.kps_time += elapsed_time

        # 计算embedding
        i=0
        for kps in kpss_list:
            face_idx = img_name+f"_{i}"
            start_time = time.time()
            face_emb, cropped_image = self.model.run_recognize(ts_img, kps)
            elapsed_time = time.time() - start_time
            logging.debug(f"embedding cost time {elapsed_time}")
            self.embed_time += elapsed_time
            self.faces[face_idx] = [file_path, face_emb, kps]
            i += 1

    def process_face_export(self, face_in_path):
        logging.info(f"Start...")
        # [(file_idx_name, file_path)]
        task_list = utils.prepare_face_set(face_in_path)
        for img_name, img_path in task_list:
            self.process_one_img(img_name, img_path)

    def process_face_similarity(self):
        logging.info(f"Start...")
        if not self.faces:
            logging.error(f"No Faces")
            return

        start_time = time.time()
        # 初始化有序的 face_idx 列表和对应的嵌入向量列表
        ordered_face_indices = []
        ordered_face_embeddings = []
        for face_idx, (file_path, face_emb, kps) in self.faces.items():
            ordered_face_indices.append(face_idx)
            ordered_face_embeddings.append(face_emb)

        # 提取所有嵌入向量
        embeddings = torch.tensor(np.stack(ordered_face_embeddings))
        # 归一化嵌入向量
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # 计算余弦相似度矩阵
        self.similarity_matrix = torch.mm(embeddings, embeddings.T)
        # 创建二进制相似度矩阵
        binary_similarity_matrix = torch.where(self.similarity_matrix > self.config["similarity_threshold"], 1, 0)

        # 初始化 parent 数组
        n = len(ordered_face_indices)
        parent = list(range(n))
        rank = [1] * n  # 使用局部变量 rank

        # 定义并查集的 find 和 union 函数
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])  # 路径压缩
            return parent[u]

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                # 按秩合并
                if rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                else:
                    parent[root_v] = root_u
                    if rank[root_u] == rank[root_v]:
                        rank[root_u] += 1

        # 遍历二进制相似度矩阵，执行 union 操作
        for i in range(n):
            for j in range(i + 1, n):
                if binary_similarity_matrix[i, j]:
                    union(i, j)

        # 根据 parent 数组构建 cluster 字典
        self.cluster = {}
        for idx_in_parent, face_idx in enumerate(ordered_face_indices):
            root = find(idx_in_parent)  # 找到根节点
            self.cluster.setdefault(root, []).append(face_idx)

        elapsed_time = time.time() - start_time
        self.similarity_time += elapsed_time

    def export_faces_cluster(self, output_path):
        logging.info(f"Start...")
        if not self.cluster:
            logging.error(f"Cluster Empty!")
            return

        start_time = time.time()
        for c_id, f_id_list in self.cluster.items():
            if len(f_id_list) < self.config["export_cluster_min_len"]:
                cluster_path = utils.os.path.join(output_path, "Others")
            else:
                cluster_path = utils.os.path.join(output_path, "Cluster_"+str(c_id)+"("+str(len(f_id_list))+")")

            utils.os.makedirs(cluster_path, exist_ok=True)
            for f_id in f_id_list:
                out_file = utils.os.path.join(cluster_path, f_id+".jpg")
                src_img_file = self.faces[f_id][0]
                src_img = cv2.imread(src_img_file)
                t_img = utils.img_2_tensor(src_img)
                out_img = utils.affine_transform(t_img, self.faces[f_id][2], self.config['exp_face_size'], self.config['exp_face_scale_factor'], self.config['exp_face_z_fix'])
                #out_img = affine_transform_standard(t_img, self.faces[f_id][2])
                out_img = out_img.permute(1, 2, 0).cpu().numpy()
                cv2.imwrite(out_file, out_img)

        elapsed_time = time.time() - start_time
        self.export_time += elapsed_time