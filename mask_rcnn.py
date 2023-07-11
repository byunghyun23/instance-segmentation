import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random


class InstanceSegmentation:
    def __init__(self):
        self.COCO_CLASS_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Load a pre-trained Mask R-CNN model
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def __get_coloured_mask(self, mask):
        colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
                   [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        r[mask == 1], g[mask == 1], b[mask == 1] = colors[random.randrange(0, 10)]
        colored_mask = np.stack([r, g, b], axis=2)

        return colored_mask

    def __prediction(self, img_path, confidence):
        img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_class = [self.COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        masks = masks[:pred_t + 1]
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]

        return masks, pred_boxes, pred_class

    def generate(self, img_path, confidence=0.5, rect_th=2, text_size=1.5, text_th=2):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks, boxes, pred_cls = self.__prediction(img_path, confidence)

        for i in range(len(masks)):
            rgb_mask = self.__get_coloured_mask(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(img, tuple(map(round, boxes[i][0])), tuple(map(round, boxes[i][1])), color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], tuple(map(round, (boxes[i][0][0], boxes[i][0][1] - 10))), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

        return img


segmentation = InstanceSegmentation()

image_path = 'image.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

instance_segmentation = segmentation.generate(image_path, confidence=0.7)

# Visualize the input image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Input Image')
plt.axis('off')

# Visualize the predicted segmentation mask
plt.subplot(1, 2, 2)
plt.imshow(instance_segmentation)
plt.title('Instance Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()
