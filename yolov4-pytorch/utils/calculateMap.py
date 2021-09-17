from __future__ import division

import math
import os
import time

import glob
import json
import os
import shutil
import operator
import sys

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.ops import nms

from copy import copy
from pathlib import Path

from utils.config import Config

from nets.yolo4 import YoloBody
from nets.yolo_training import Generator, YOLOLoss

from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import non_max_suppression, bbox_iou, DecodeBox,letterbox_image,yolo_correct_boxes

# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('svg')  # for writing to files only

def centerxywhYoloTox1y1x2y2(width, height, xCenter, yCenter, widthPer, heightPer):
    x = xCenter * width
    y = yCenter * height

    return x - (width * widthPer)/2, y - (height * heightPer)/2, x + (width * widthPer) /2, y + (height * heightPer)/2

def storeGtDetectRes(epochFolder, images_val, targets_val, outputs, filePaths):
    batchSize = outputs[0].shape[0]

    gtFolder = epochFolder + "/gt/"
    drFolder = epochFolder + "/dr/"
    if not os.path.exists(gtFolder):
        os.makedirs(gtFolder)
    if not os.path.exists(drFolder):
        os.makedirs(drFolder)

    for iCount in range(batchSize):
        outputsOnlyOne = []
        filePath = filePaths[iCount].split(" ")[0]
        #print("Handling image ", filePath)
        image_valImg = images_val[iCount:iCount+1]
        targets_valImg = targets_val[iCount]
        outputsOnlyOne.append(outputs[0][iCount:iCount+1])
        outputsOnlyOne.append(outputs[1][iCount:iCount+1])
        outputsOnlyOne.append(outputs[2][iCount:iCount+1])

        handleEachImageDrGt(epochFolder, image_valImg, targets_valImg, outputsOnlyOne, filePath)

def handleEachImageDrGt(epochFolder, image_valImg, targets_valImg, outputsImg, filePath):
    gtFolder = epochFolder + "/gt/"
    drFolder = epochFolder + "/dr/"
    baseName = os.path.basename(filePath).split(".")[0]

    imageSize = image_valImg.shape
    width = int(imageSize[2])
    height = int(imageSize[3])
    #print("image Size", imageSize)

    #Write gt file lines
    gt_file = open(gtFolder + baseName + ".txt", 'w')
    #print("target img ", targets_valImg[0][0])
    #print("target img ", targets_valImg[0][1])
    #0.5361,  0.4519,  0.3510,  0.5000, 14.0000
    for iTargetsCount in range(targets_valImg.shape[0]):
        x1,y1,x2,y2 = centerxywhYoloTox1y1x2y2(width, height,
                                               float(targets_valImg[iTargetsCount][0]), float(targets_valImg[iTargetsCount][1]),
                                               float(targets_valImg[iTargetsCount][2]), float(targets_valImg[iTargetsCount][3]))
        targetLineStr = str(int(targets_valImg[iTargetsCount][4])) + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2)
        gt_file.write('%s ' % (targetLineStr))
        gt_file.write('\n')
    gt_file.close()

    dr_file = open(drFolder + baseName + ".txt", 'w')

    yolo_decodes = []
    config = Config

    model_image_size = (width,height, 3)
    image_shape = np.array([width,height])
    confidence = 0.01
    iou = 0.5
    for i in range(3):
        yolo_decodes.append(DecodeBox(config["yolo"]["anchors"][i], config["yolo"]["classes"],
                                           (model_image_size[1], model_image_size[0])))

    output_list = []
    print("outputsImg[0] ", outputsImg[0].shape, " ", outputsImg[1].shape, " ", outputsImg[2].shape)
    for i in range(3):
        output_list.append(yolo_decodes[i](outputsImg[i]))

    output = torch.cat(output_list, 1)

    batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                           conf_thres=confidence,
                                           nms_thres=iou)
    #print("batch_detections Got ", len(batch_detections))
    try:
        batch_detections = batch_detections[0].cpu().numpy()
    except:
        print("Exception ", len(batch_detections))
        return

    top_index = batch_detections[:, 4] * batch_detections[:, 5] > confidence
    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
    top_label = np.array(batch_detections[top_index, -1], np.int32)
    top_bboxes = np.array(batch_detections[top_index, :4])
    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1),\
                                             np.expand_dims(top_bboxes[:, 1],-1), \
                                             np.expand_dims(top_bboxes[:, 2], -1), \
                                             np.expand_dims(top_bboxes[:, 3], -1)
    # 去掉灰条
    boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                               np.array([model_image_size[0], model_image_size[1]]), image_shape)

    #print("boxes ", boxes.shape)

    for i, c in enumerate(top_label):
        #predicted_class = class_names[c]
        score = str(top_conf[i])

        top, left, bottom, right = boxes[i]
        lineItem = str(c) + " " + score[:6] + " " + str(int(left)) \
                   + " "  + str(int(top)) +  " " + str(int(right)) + " " + str(int(bottom))
        #print("lineItem:", lineItem)
        dr_file.write('%s ' % (lineItem))
        dr_file.write('\n')
    dr_file.close()

"""
 throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def calculateMAP(GT_PATH, DR_PATH, results_files_path):
    MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

    '''
        0,0 ------> x (width)
         |
         |  (Left,Top)
         |      *_________
         |      |         |
                |         |
         y      |_________|
      (height)            *
                    (Right,Bottom)
    '''

    # make sure that the cwd() is the location of the python script (so that every path makes sense)
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # if there are no images then no animation can be shown

    """
     Create a ".temp_files/" and "results/" directory
    """
    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)

    if os.path.exists(results_files_path): # if it exist already
        # reset the results directory
        shutil.rmtree(results_files_path)

    os.makedirs(results_files_path)

    """
     ground-truth
         Load each of the ground-truth files into a temporary ".json" file.
         Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        #print(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                        class_name, left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                else:
                        class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            # check if class is in the ignore list, if yes skip
            #if class_name in ignore:
            #    continue
            bbox = left + " " + top + " " + right + " " +bottom
            if is_difficult:
                    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                    is_difficult = False
            else:
                    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                    # count that object
                    if class_name in gt_counter_per_class:
                        gt_counter_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        gt_counter_per_class[class_name] = 1

                    if class_name not in already_seen_classes:
                        if class_name in counter_images_per_class:
                            counter_images_per_class[class_name] += 1
                        else:
                            # if class didn't exist yet
                            counter_images_per_class[class_name] = 1
                        already_seen_classes.append(class_name)


        # dump bounding_boxes into a ".json" file
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    #print(gt_classes)
    #print(gt_counter_per_class)

    """
     detection-results
         Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            #print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    #print("match")
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
                    #print(bounding_boxes)
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    sum_Prec = 0.0
    sum_Rec = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # open file to store the results
    with open("./logs/mapDetailedResults.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}

        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
             Load detection-results of that class
            """
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            """
             Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            score = [0] * nd
            score05_idx = 0
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                score[idx]   = float(detection["confidence"])
                if score[idx] > 0.5:
                    score05_idx = idx

                # assign detection-results to ground truth object if any
                # open ground-truth with that file_id
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [ float(x) for x in detection["bbox"].split() ]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [ float(x) for x in obj["bbox"].split() ]
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign detection as true positive/don't care/false positive
                # set minimum overlap
                min_overlap = MINOVERLAP

                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                            if not bool(gt_match["used"]):
                                # true positive
                                tp[idx] = 1
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                # update the ".json" file
                                with open(gt_file, 'w') as f:
                                        f.write(json.dumps(ground_truth_data))
                            else:
                                # false positive (multiple detection)
                                fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            #print(tp)
            rec = tp[:]

            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print(prec)
            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            F1 = np.array(rec)*np.array(prec)/(np.array(prec)+np.array(rec))*2

            sum_AP += ap
            sum_Prec += np.mean(mrec)
            sum_Rec += np.mean(mprec)
            text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)

            if len(prec)>0:
                F1_text = "{0:.2f}".format(F1[score05_idx]) + " = " + class_name + " F1 "
                Recall_text = "{0:.2f}%".format(rec[score05_idx]*100) + " = " + class_name + " Recall "
                Precision_text = "{0:.2f}%".format(prec[score05_idx]*100) + " = " + class_name + " Precision "
            else:
                F1_text = "0.00" + " = " + class_name + " F1 "
                Recall_text = "0.00%" + " = " + class_name + " Recall "
                Precision_text = "0.00%" + " = " + class_name + " Precision "
            """
             Write to results.txt
            """
            rounded_prec = [ '%.2f' % elem for elem in prec ]
            rounded_rec = [ '%.2f' % elem for elem in rec ]
            #print("F1 ", F1, " rec ", rec, " prec ", prec)
            if (len(F1) == 0 or len(rec)==0 or len(prec) ==0):
                results_file.write(" F1 rec prec all 0")
            else:
                results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n")
                results_file.write(text + "\t||\tscore_threhold=0.5 : " + "F1=" + "{0:.2f}".format(F1[score05_idx])\
                         + " ; Recall=" + "{0:.2f}%".format(rec[score05_idx]*100) + " ; Precision=" + "{0:.2f}%".format(prec[score05_idx]*100))
            results_file.write("\n\n")
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]

        results_file.write("\n# mAP of all classes\n")
        aPrec = sum_Prec / n_classes
        aRec = sum_Rec / n_classes

        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)

    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)

    return aPrec, aRec, mAP

#To be modified
def plot_results(start=0, stop=0, bucket='', id=(), labels=(), save_dir='./logs'):
    # Plot training 'results*.txt'. from utils.plots import *; plot_results(save_dir='runs/train/exp')
    fig, ax = plt.subplots(1, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['epoch', 'mAP', 'valLoss', 'precision','recall']

    files = glob.glob(save_dir + '/mapLogs*.txt')
    #print("files ", files)
    assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)
    for fi, f in enumerate(files):
        try:
            print("each file  ", f)
            results = []
            lines = [fileLine.strip() for fileLine in open(f).readlines()]
            #print("lines ", lines)
            metricsData = []
            for eachLine in lines:
                splitParts = eachLine.split(" ")
                rowData = []
                for eachItem in splitParts:
                    num = float(eachItem.split(":")[1])
                    rowData.append(num)
                metricsData.append(np.array(rowData))
            results = np.array(metricsData)
            #print("before T", results)

            results = results.T
            #print("after T", results)
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(5):
                y = results[i, x]
                #print("y value:", y)
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else '' #os.path.basename(f)
                #print("Going to plot ", x, " ", y)
                ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    fig.savefig(Path(save_dir) / 'lossMapResults.png', dpi=200)

if __name__ == "__main__":
    print("This is the test of main methods")
    plot_results(save_dir='../logs')









