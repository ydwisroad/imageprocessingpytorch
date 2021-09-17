import cv2
import os
import xml.etree.cElementTree as ET

train_xml_dir = '/home/zj/data/PASCAL-VOC/2007/train/Annotations'
train_jpeg_dir = '/home/zj/data/PASCAL-VOC/2007/train/JPEGImages'

test_xml_dir = '/home/zj/data/PASCAL-VOC/2007/test/Annotations'
test_jpeg_dir = '/home/zj/data/PASCAL-VOC/2007/test/JPEGImages'

# 标注图像保存路径
train_imgs_dir = '/home/zj/data/PASCAL-VOC/2007/train_imgs'
test_imgs_dir = '/home/zj/data/PASCAL-VOC/2007/test_imgs'


def parse_xml(xml_path):
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()

    img_name = ''
    obj_list = list()
    bndbox_list = list()

    # 遍历根节点下所有节点，查询文件名和目标坐标
    for child_node in root:
        if 'filename'.__eq__(child_node.tag):
            img_name = child_node.text
        if 'object'.__eq__(child_node.tag):
            obj_name = ''
            for obj_node in child_node:
                if 'name'.__eq__(obj_node.tag):
                    obj_name = obj_node.text
                if 'bndbox'.__eq__(obj_node.tag):
                    node_bndbox = obj_node

                    node_xmin = node_bndbox[0]
                    node_ymin = node_bndbox[1]
                    node_xmax = node_bndbox[2]
                    node_ymax = node_bndbox[3]

                    obj_list.append(obj_name)
                    bndbox_list.append((
                        int(node_xmin.text), int(node_ymin.text), int(node_xmax.text), int(node_ymax.text)))

    return img_name, obj_list, bndbox_list


def batch_parse(xml_dir, jpeg_dir, imgs_dir):
    xml_list = os.listdir(xml_dir)
    jepg_list = os.listdir(jpeg_dir)

    for xml_name in xml_list:
        xml_path = os.path.join(xml_dir, xml_name)
        img_name, obj_list, bndbox_list = parse_xml(xml_path)
        print(img_name, obj_list, bndbox_list)

        if img_name in jepg_list:
            img_path = os.path.join(jpeg_dir, img_name)
            src = cv2.imread(img_path)
            for i in range(len(obj_list)):
                obj_name = obj_list[i]
                bndbox = bndbox_list[i]

                obj_dir = os.path.join(imgs_dir, obj_name)
                if not os.path.exists(obj_dir):
                    os.mkdir(obj_dir)
                obj_path = os.path.join(obj_dir, '%s-%s-%d-%d-%d-%d.png' % (
                    img_name, obj_name, bndbox[0], bndbox[1], bndbox[2], bndbox[3]))

                res = src[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]
                cv2.imwrite(obj_path, res)


if __name__ == '__main__':
    batch_parse(train_xml_dir, train_jpeg_dir, train_imgs_dir)
    batch_parse(test_xml_dir, test_jpeg_dir, test_imgs_dir)