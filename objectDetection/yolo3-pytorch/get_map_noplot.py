import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math

import numpy as np
#----------------------------------------------------#
#   用于计算mAP
#   代码克隆自https://github.com/Cartucho/mAP

#Another good one is: https://github.com/rafaelpadilla/Object-Detection-Metrics
#----------------------------------------------------#

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


def calculateMAP(gtPath, drPath, results_files_path, epoch):
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
    ap_dictionary = {}
    lamr_dictionary = {}
    # open file to store the results
    with open(results_files_path + "/results_" + str(epoch) + " .txt", 'w') as results_file:
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
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n")
            results_file.write(text + "\t||\tscore_threhold=0.5 : " + "F1=" + "{0:.2f}".format(F1[score05_idx])\
                     + " ; Recall=" + "{0:.2f}%".format(rec[score05_idx]*100) + " ; Precision=" + "{0:.2f}%".format(prec[score05_idx]*100))
            results_file.write("\n\n")
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP*100)
        results_file.write(text + "\n")
        print(text)

    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)

    return mAP

if __name__ == "__main__":
    GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth')
    DR_PATH = os.path.join(os.getcwd(), 'input', 'detection-results')

    results_files_path = "results"
    finalMAP = calculateMAP(GT_PATH, DR_PATH, results_files_path, 2)
    print("finalMaP ", finalMAP)


