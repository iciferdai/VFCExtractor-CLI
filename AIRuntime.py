from ConfigLoader import logging
import Utils as utils
from Utils import torch
from Utils import trans
from Utils import v2
import onnxruntime
import numpy as np


class Models:
    def __init__(self):
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device='cuda:0')
        self.config = utils.G_CONFIG
        self.retinaface_model_path = './models/det_10g.onnx'
        self.retinaface_model = onnxruntime.InferenceSession(self.retinaface_model_path, providers=self.providers)
        self.recognition_model_path = './models/w600k_r50.onnx'
        self.recognition_model = onnxruntime.InferenceSession(self.recognition_model_path, providers=self.providers)
        self.arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)

    def detect_retinaface(self, img, max_num, score):

        # Resize image to fit within the input_size
        input_size = (640, 640)
        im_ratio = torch.div(img.size()[1], img.size()[2])

        # model_ratio = float(input_size[1]) / input_size[0]
        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height, img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1, 2, 0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
        det_img[:new_height, :new_width, :] = img

        # Switch to BGR and normalize
        det_img = det_img[:, :, [2, 1, 0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1)  # 3,128,128

        # Prepare data and find model parameters
        det_img = torch.unsqueeze(det_img, 0).contiguous()

        io_binding = self.retinaface_model.io_binding()
        io_binding.bind_input(name='input.1', device_type='cuda', device_id=0, element_type=np.float32,
                              shape=det_img.size(), buffer_ptr=det_img.data_ptr())

        io_binding.bind_output('448', 'cuda')
        io_binding.bind_output('471', 'cuda')
        io_binding.bind_output('494', 'cuda')
        io_binding.bind_output('451', 'cuda')
        io_binding.bind_output('474', 'cuda')
        io_binding.bind_output('497', 'cuda')
        io_binding.bind_output('454', 'cuda')
        io_binding.bind_output('477', 'cuda')
        io_binding.bind_output('500', 'cuda')

        # Sync and run model
        self.syncvec.cpu()
        self.retinaface_model.run_with_iobinding(io_binding)

        net_outs = io_binding.copy_outputs_to_cpu()

        input_height = det_img.shape[2]
        input_width = det_img.shape[3]

        fmc = 3
        center_cache = {}
        scores_list = []
        bboxes_list = []
        kpss_list = []
        for idx, stride in enumerate([8, 16, 32]):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride

            kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = np.stack([anchor_centers] * 2, axis=1).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= score)[0]

            x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
            y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
            x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
            y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

            bboxes = np.stack([x1, y1, x2, y2], axis=-1)

            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            preds = []
            for i in range(0, kps_preds.shape[1], 2):
                px = anchor_centers[:, i % 2] + kps_preds[:, i]
                py = anchor_centers[:, i % 2 + 1] + kps_preds[:, i + 1]

                preds.append(px)
                preds.append(py)
            kpss = np.stack(preds, axis=-1)
            # kpss = kps_preds
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()  ###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        dets = pre_det
        thresh = 0.4
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scoresb = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        orderb = scoresb.argsort()[::-1]

        keep = []
        while orderb.size > 0:
            i = orderb[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[orderb[1:]])
            yy1 = np.maximum(y1[i], y1[orderb[1:]])
            xx2 = np.minimum(x2[i], x2[orderb[1:]])
            yy2 = np.minimum(y2[i], y2[orderb[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            orderb = orderb[inds + 1]

        det = pre_det[keep, :]

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            det_img_center = det_img.shape[0] // 2, det_img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            if kpss is not None:
                kpss = kpss[bindex, :]

        return kpss

    def recognize(self, img, face_kps):
        # Find transform
        dst = self.arcface_dst.copy()
        dst[:, 0] += 8.0

        tform = trans.SimilarityTransform()
        tform.estimate(face_kps, dst)

        # Transform
        img = v2.functional.affine(img, tform.rotation * 57.2958, (tform.translation[0], tform.translation[1]),
                                   tform.scale, 0, center=(0, 0))
        img = v2.functional.crop(img, 0, 0, 128, 128)

        img = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(img)

        # Switch to BGR and normalize
        img = img.permute(1, 2, 0)  # 112,112,3
        cropped_image = img
        img = img[:, :, [2, 1, 0]]
        img = torch.sub(img, 127.5)
        img = torch.div(img, 127.5)
        img = img.permute(2, 0, 1)  # 3,112,112

        # Prepare data and find model parameters
        img = torch.unsqueeze(img, 0).contiguous()
        input_name = self.recognition_model.get_inputs()[0].name

        outputs = self.recognition_model.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = self.recognition_model.io_binding()
        io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=np.float32,
                              shape=img.size(), buffer_ptr=img.data_ptr())

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], 'cuda')

        # Sync and run model
        #self.syncvec.cpu()
        self.recognition_model.run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image

    def run_detect(self, img, detect_mode='Retinaface', max_num=1, score=0.5):
        kpss = self.detect_retinaface(img, max_num=max_num, score=score)
        return kpss

    def run_recognize(self, img, kps):
        embedding, cropped_image = self.recognize(img, kps)
        return embedding, cropped_image