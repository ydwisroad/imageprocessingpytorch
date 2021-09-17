

import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    A.CenterCrop(width=1080, height=1080)
],
   bbox_params=A.BboxParams(format='pascal_voc')
)

class_labels = ['bag']

#https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("/Users/i052090/Downloads/segmentation/data/bagspic/marked/original/1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

bboxes = [
    [123, 174, 795, 988, 'bag']
]

# Augment an image
transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
transformed_image = transformed["image"]
transformed_bbox = transformed['bboxes']
cv2.imwrite("./cropcenter.png", transformed_image)

print("transformed bbox ", transformed_bbox)



