# 通过onnx推理

import onnx
import onnxruntime as ort
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from typing import List

class YOLO_ONNX:
    def __init__(self, model_path:str = "./best.onnx"):
        self.model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # 预先计算目标尺寸
        self.dst_size = (640, 640)
        self.dst_w, self.dst_h = self.dst_size

    def preprocess_batch(self, images: List[np.ndarray]):
        """优化的批量预处理"""
        batch_data = []
        transforms = []
        
        for image in images:
            # 计算缩放比例和偏移量
            scale = min(self.dst_w / image.shape[1], self.dst_h / image.shape[0])
            ox = (self.dst_w - scale * image.shape[1]) / 2
            oy = (self.dst_h - scale * image.shape[0]) / 2
            
            # 创建变换矩阵
            M = np.array([[scale, 0, ox], [0, scale, oy]], dtype=np.float32)
            IM = cv2.invertAffineTransform(M)
            
            # 仿射变换（使用常量边界填充）
            img_pre = cv2.warpAffine(image, M, self.dst_size, 
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(114, 114, 114))
            
            # 一次性完成归一化和转置操作
            img_pre = img_pre[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            
            batch_data.append(img_pre)
            transforms.append(IM)
            
        return np.stack(batch_data), transforms

    def postprocess_batch(self, results, original_images, transforms):
        """优化的批量后处理"""
        batch_size = len(original_images)
        masks_list = []
        
        # 批量处理检测结果
        box_results = results[0].transpose(0, 2, 1)  # 批量转置
        seg_results = results[1]  # 批量分割结果
        
        for i in range(batch_size):
            # 处理单个图像的检测框
            box_result = box_results[i:i+1]
            box_dets = self.box_process(box_result)
            box_dets = torch.from_numpy(np.array(box_dets).reshape(-1, 38))
            
            if len(box_dets) == 0:
                # 如果没有检测到目标，返回空白掩码
                h, w = original_images[i].shape[:2]
                masks_list.append(np.zeros((h, w), dtype=np.uint8))
                continue
            
            # 处理分割结果
            seg_result = seg_results[i]
            mask_results = self.mask_process(
                seg_result,
                box_dets[:, 6:],
                box_dets[:, :4],
                self.dst_size,
                upsample=True
            )
            
            # 转换回原始图像尺寸
            h, w = original_images[i].shape[:2]
            instance_mask = self.draw_instance_masks(
                original_images[i], 
                transforms[i], 
                mask_results
            )
            masks_list.append(instance_mask)
            
        return masks_list

    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        真正的批量处理图像
        Args:
            images: 图像列表，每个元素都是numpy数组
        Returns:
            mask列表
        """
        if not images:
            return []

        try:
            # 1. 批量预处理
            batch_input, transforms = self.preprocess_batch(images)
            
            # 2. 批量推理 - 一次性处理所有图片
            results = self.model.run(None, {'images': batch_input})
            
            # 3. 批量后处理
            return self.postprocess_batch(results, images, transforms)
            
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            # 如果批处理失败，回退到逐个处理
            masks = []
            for image in images:
                try:
                    mask = self.predict(image)
                    masks.append(mask)
                except Exception as e:
                    print(f"Error processing single image: {str(e)}")
                    h, w = image.shape[:2]
                    masks.append(np.zeros((h, w), dtype=np.uint8))
            return masks

    def predict(self, image: np.ndarray) -> np.ndarray:
        """优化后的单图预测"""
        # 使用批处理代码处理单张图片
        batch_input, transforms = self.preprocess_batch([image])
        results = self.model.run(None, {'images': batch_input})
        masks = self.postprocess_batch(results, [image], transforms)
        return masks[0]

    def draw_instance_masks(self, image, IM, mask_results):
        """优化的实例掩码绘制"""
        h, w = image.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(mask_results) == 0:
            return instance_mask
            
        # 批量处理所有掩码
        for i, mask in enumerate(mask_results, 1):
            mask = mask.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.warpAffine(mask, IM, (w, h),
                                        flags=cv2.INTER_LINEAR)
            instance_mask[mask_resized == 1] = i
            
        return instance_mask

    # 前处理
    def pre_process(self, image: np.ndarray)->np.ndarray:
        image = cv2.resize(image, (640, 640))
        cv2.imwrite("temp.bmp", image)
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis, ...].astype(np.float32)
        return image

    # 仿射变换缩放
    def preprocess_warpAffine(self , image, dst_width=640, dst_height=640):
        scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
        ox = (dst_width - scale * image.shape[1]) / 2
        oy = (dst_height - scale * image.shape[0]) / 2
        M = np.array([
            [scale, 0, ox],
            [0, scale, oy]
        ], dtype=np.float32)

        img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        IM = cv2.invertAffineTransform(M)

        img_pre = (img_pre[..., ::-1] / 255.0).astype(np.float32)
        img_pre = img_pre.transpose(2, 0, 1)[None]
        img_pre = torch.from_numpy(img_pre)
        img_pre = img_pre.numpy()
        return img_pre, IM

    # NMS
    def NMS(self, boxes, iou_thres):
        remove_flags = [False] * len(boxes)
        keep_boxes = []
        for i, ibox in enumerate(boxes):
            if remove_flags[i]:
                continue
            keep_boxes.append(ibox)
            for j in range(i + 1, len(boxes)):
                if remove_flags[j]:
                    continue
                jbox = boxes[j]
                if ibox[5] != jbox[5]:
                    continue
                if self.iou(ibox, jbox) > iou_thres:
                    remove_flags[j] = True
        return keep_boxes

    # 计算面积
    def area_box(self , box):
        return (box[2] - box[0]) * (box[3] - box[1])

    # iou
    def iou(self, box1, box2):


        left = max(box1[0], box2[0])
        top = max(box1[1], box2[1])
        right = min(box1[2], box2[2])
        bottom = min(box1[3], box2[3])
        cross = max((right - left), 0) * max((bottom - top), 0)
        union = self.area_box(box1) + self.area_box(box2) - cross
        if cross == 0 or union == 0:
            return 0
        return cross / union

    # 后处理
    def box_process(self, pred, conf_thres=0.25, iou_thres=0.45) -> list:

        boxes = []
        for item in pred[0]:
            # print(item.shape)
            cx, cy, w, h = item[:4]
            label = item[4:-32].argmax()
            confidence = item[4 + label]
            if confidence < conf_thres:
                continue
            left = cx - w * 0.5
            top = cy - h * 0.5
            right = cx + w * 0.5
            bottom = cy + h * 0.5
            boxes.append([left, top, right, bottom, confidence, label, *item[-32:]]) # 自我约定，多加了一个label, 所以是38

        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

        return self.NMS(boxes, iou_thres)

    # 裁剪mask
    def crop_mask(self ,masks, boxes):

        # masks -> n, 160, 160  原始 masks
        # boxes -> n, 4         检测框，映射到 160x160 尺寸下的
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))



    # 分割结果处理
    def mask_process(self , protos, masks_in, bboxes, shape, upsample=False):

        # protos   -> 32, 160, 160 分割头输出
        # masks_in -> n, 32        检测头输出的 32 维向量，可以理解为 mask 的权重
        # bboxes   -> n, 4         检测框
        # shape    -> 640, 640     输入网络中的图像 shape
        # unsample 一个 bool 值，表示是否需要上采样 masks 到图像的原始形状

        # 将 numpy.ndarray 转换为 PyTorch 张量
        protos = torch.from_numpy(protos)
        # masks_in = torch.from_numpy(masks_in) # 这里不需要转换，因为 masks_in 已经是 torch.Tensor 类型

        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        # 矩阵相乘 nx32 @ 32x(160x160) -> nx(160x160) -> sigmoid -> nx160x160
        masks = (masks_in.float() @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

        downsampled_bboxes = bboxes.clone()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        if upsample:
            masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        return masks.gt_(0.25) # 值大于0.5的为True，小于0.5的为False

    # 随机颜色
    def random_color(self , id):
        h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
        s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
        return self.hsv2bgr(h_plane, s_plane, 1)

    # hsv2bgr
    def hsv2bgr(self , h, s, v):
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        r, g, b = 0, 0, 0

        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        elif h_i == 5:
            r, g, b = v, p, q

        return int(b * 255), int(g * 255), int(r * 255)

    # 绘制实例mask
    def draw_instance_masks(self , image, IM, mask_results):
        h, w = image.shape[:2]
        instance_mask = np.zeros((h, w), dtype=np.uint8)

        for i, mask in enumerate(mask_results):
            mask = mask.cpu().numpy().astype(np.uint8)  # 640x640
            mask_resized = cv2.warpAffine(mask, IM, (w, h),
                                          flags=cv2.INTER_LINEAR)  # Resize mask to original image size

            instance_mask[mask_resized == 1] = i + 1  # Assign instance number to mask

        return instance_mask



    # 绘制所有
    def draw_all(self, img, IM, box_results, mask_results, names):
        boxes = np.array(box_results[:, :6])
        lr = boxes[:, [0, 2]]
        tb = boxes[:, [1, 3]]
        boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]
        boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]

        # draw mask
        h, w = img.shape[:2]
        for i, mask in enumerate(mask_results):
            mask = mask.cpu().numpy().astype(np.uint8)  # 640x640
            mask_resized = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR)  # 1080x810

            label = int(boxes[i][5])
            color = np.array(self.random_color(label))

            colored_mask = (np.ones((h, w, 3)) * color).astype(np.uint8)
            masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)

            mask_indices = mask_resized == 1
            img[mask_indices] = (img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)

            # contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img, contours, -1, random_color(label), 2)

        # draw box
        for obj in boxes:
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = self.random_color(label)
            cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
            caption = f"{names[label]} {confidence:.2f}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
            cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

        cv2.imwrite("infer-seg.jpg", img)







    # 推理
    def predict(self, image: np.ndarray)->list:
        # 确保传入的image转为了numpy.ndarray
        image = np.asarray(image)
        # image = self.pre_process(image)
        # 备份一份可以用于绘制的image
        ori_image = image.copy()
        image, IM = self.preprocess_warpAffine(image)
        # print(image.shape) # torch.Size([1, 3, 640, 640])
        # 转换为 numpy.ndarray
        # image = image.numpy()
        # cv2.imshow("preprocess", image[0].transpose(1, 2, 0))
        # cv2.waitKey(0)
        results = self.model.run(None, {'images': image})
        # print(len(results))
        # print(results[0].shape) # (1,37,8400) ,4是box ， 1是一个类的conf , 32 维的向量可以看作是与每个检测框关联的分割 mask 的系数或权重
        # print(results[1].shape) # (1,32,160,160)
        # 把box_result转换为(1,8400,37)
        box_result = results[0].transpose(0, 2, 1)
        # print(box_result.shape)
        # 把seg_result转换为(32,160,160)
        seg_result = results[1][0]
        # print(seg_result.shape)
        # output1 = results[1][2][0]  # 32,160,160 分割头输出
        box_results = self.box_process(box_result)
        box_results = torch.from_numpy(np.array(box_results).reshape(-1, 38)) # 后文的pred对应的是这个
        # print(results.shape)
        mask_results = self.mask_process(seg_result, box_results[:, 6:], box_results[:, :4], (640, 640), upsample=True) # 后文的masks对应的是这个
        # print(mask_results.shape)
        # print(mask_results)
        # self.draw_all(ori_image , IM ,box_results , mask_results , ['0'])
        mask_image = self.draw_instance_masks(ori_image, IM, mask_results)
        # cv2.imwrite("mask_image_py.bmp", mask_image)
        # print(mask_image.shape)


        return mask_image

# # 加载模型
# model_path = r'D:\Gold_wire\code\weights\goldwire_complete.onnx'
# model = onnx.load(model_path)
# # 创建一个运行时
# ort_session = ort.InferenceSession(model_path)
# # 查看模型输入
# for input in ort_session.get_inputs():
#     print(input)
# # 查看模型输出
# for output in ort_session.get_outputs():
#     print(output)
# # 创建一个与模型输入兼容的虚拟输入张量
# import numpy as np
# dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
# # 运行模型
# outputs = ort_session.run(None, {'input': dummy_input})
def model_detail(model_path):
    model = onnx.load(model_path)
    # 查看模型输入
    for input in model.graph.input:
        print(input)

    # 查看模型输出
    for output in model.graph.output:
        print(output)

def model_infer(model_path, image):
    ort_session = ort.InferenceSession(model_path)

    # Preprocess the image to match the input shape and type
    # Assuming `image` is a NumPy array with shape (640, 640, 3)
    image = cv2.resize(image, (640, 640))  # Resize to (640, 640)
    image = image.transpose(2, 0, 1)  # Change shape to (3, 640, 640)
    image = image[np.newaxis, ...].astype(np.float32)  # Add batch dimension and convert to float32

    # Run the model
    outputs = ort_session.run(None, {'images': image})
    return outputs

if __name__ == "__main__":
    model_path = r'D:\Gold_wire\code\weights\yolov8l-seg-640-origintype-3000.onnx'
    image_path = r'D:\Gold_wire\code\data\data3\gray-bmp_all\baseView00005.bmp'
    image = cv2.imread(image_path)
    model = YOLO_ONNX(model_path)
    results = model.predict(image)

    print(len(results[1]))
    # model_detail(model_path)
