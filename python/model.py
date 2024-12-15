# model.py
import cv2 
from ultralytics import YOLO
import numpy as np

class MODEL:
    def __init__(self, model_path:str = "./best.pt"):
        self.model = YOLO(model_path)

    def predict(self, image: np.ndarray) -> list:
        image = np.asarray(image)
        image = cv2.resize(image, (640, 640))
        cv2.imwrite("save.bmp",image)
        
        # results = self.model.predict(image, conf=0.25)
        # img_width = image.shape[1]
        # img_height = image.shape[0]
        sorted_result = []
        # for result in results:
        #     if result.masks is None or len(result.masks) == 0:
        #         continue
        #     for mask in result.masks:
        #         class_name = '0'
        #         data = mask.xy
        #         data = data[0].tolist()
        #         for tuple in data:
        #             x = tuple[0] / img_width
        #             y = tuple[1] / img_height
        #             class_name = class_name + ' ' + str(x) + ' ' + str(y)
        #         sorted_result.append(str(class_name))  # 确保元素为字符串
        return sorted_result