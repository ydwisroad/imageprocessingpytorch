import os
from lxml import etree
import json
import shutil
import cv2

annotation_root = "E:/roadproject/experiment/data/TrafficSign/data/annotations.json"
root_path = "E:/roadproject/experiment/data/TrafficSign/data/"

save_file_root = "./my_yolo_dataset"

def main():
    print("Start to parse json " + annotation_root)

    # read class_indict
    json_file = open(annotation_root, 'r')
    class_dict = json.load(json_file)

    print("keys:" , class_dict.keys())

    types = class_dict["types"]
    print(types.index("i10"))

    imgs = class_dict["imgs"]
    allImageKeys = imgs.keys()

    for eachKey in allImageKeys:
        print("each Key:" , eachKey)
        path = imgs[eachKey]["path"]
        objects = imgs[eachKey]["objects"]
        id = imgs[eachKey]["id"]

        list_file = open( os.path.join(root_path,'labels/', '%s.txt'%(eachKey)), 'w')

        file_img = os.path.join(root_path, path)
        img = cv2.imread(file_img)
        print("image shape ", img.shape)
        width = img.shape[0]
        height = img.shape[1]
        for eachObject in objects:
            category = eachObject["category"]
            bbox = eachObject["bbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            index = types.index(category)
            list_file.write('%s '%(index))
            xCenter = (xmax + xmin) / (2 * width)
            yCenter = (ymin + ymax ) / (2 * height)
            widthPer = (xmax - xmin) / width
            heightPer = (ymax - ymin)/ height
            list_file.write('%.6f ' % (xCenter))
            list_file.write('%.6f ' % (yCenter))
            list_file.write('%.6f ' % (widthPer))
            list_file.write('%.6f ' % (heightPer))
            list_file.write('\n')
        list_file.close()



if __name__ == "__main__":
    main()

