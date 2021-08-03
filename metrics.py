import numpy as np
import math
import xml.etree.ElementTree as ET
def iou(detection, label):
    box_2d = detection.box_2d
    detected_x = box_2d[1][0] - box_2d[0][0]
    label_x = label['Box_2D'][1][0] - label['Box_2D'][0][0]
    x = max(box_2d[1][0], label['Box_2D'][1][0]) - min(box_2d[0][0], label['Box_2D'][0][0])
    overlap_x = (detected_x + label_x - x) if (detected_x + label_x - x) >=0 else 0
    detected_y = box_2d[1][1] - box_2d[0][1]
    label_y = label['Box_2D'][1][1] - label['Box_2D'][0][1]
    y = max(box_2d[1][1], label['Box_2D'][1][1]) - min(box_2d[0][1], label['Box_2D'][0][1])
    overlap_y = (detected_y + label_y - y) if (detected_y + label_y - y) >=0 else 0
    overlap = overlap_x * overlap_y
    union = detected_x * detected_y + label_x * label_y - overlap
    return overlap/union

def recall(detections, labels):
    iou_list = np.zeros(len(detections), len(labels))
    for i,detection in enumerate(detections):
        for j,label in enumerate(labels):
            iou_list[i,j] = iou(detection, label)
    sort_list = np.argsort(-iou_list, axis=1)
    largest_overlap = np.nonzero(sort_list == 0)[1]
    assign_dict = {} # detection's corresponding label
    label_dict = {} # label's corresponding detection
    tp = 0
    fn = 0
    for detection_idx,label_idx in enumerate(largest_overlap):
        if iou_list[detection_idx, label_idx] <= 0.5:
            assign_dict[detection_idx] = -1
            fn += 1
        elif label_idx in label_dict:
            if iou_list[detection_idx][label_idx] > iou_list[label_dict[label_idx]][label_idx]:
                # problem: multiple assignment actually not assigned?
                assign_dict[label_dict[label_idx]] = -1
                label_dict[label_idx] = detection_idx
                assign_dict[detection_idx] = label_idx
            else:
                assign_dict[detection_idx] = -1
        else:
            assign_dict[detection_idx] = label_idx
            label_dict[label_idx] = detection_idx
            tp += 1
    return tp/(tp+fn), assign_dict

def orientation_similarity(orientations, labels, assign_dict):
    d = len(orientations)
    sum = 0
    for i, orient in enumerate(orientations):
        if assign_dict[i] != -1:
            sum += (1 + math.cos(orient, labels[assign_dict[i]]['Theta'])) / 2
    return 1 / d * sum

def average_orientation_similarity(orientation_sets, label_sets, detection_sets):
    recall_list = []
    assign_dict_list = []
    for i in len(detection_sets):
        recall, assign_dict = recall(detection_sets[i], label_sets[i])
        recall_list.append(recall)
        assign_dict_list.append(assign_dict)
    recall_list = np.array(recall_list)
    sr_sum = 0
    for r in range(0, 1.1, 0.1):
        indices = np.nonzero(recall_list >= r)[0]
        sr_list = []
        for i in indices:
            sr_list.append(orientation_similarity(orientation_sets[i], label_sets[i], assign_dict_list[i]))
        sr_sum += max(sr_list)
    return sr_sum / 11

def average_orientation_error(orientation_sets, label_sets):
    sum = 0
    num = 0
    for i in len(orientation_sets):
        for j in len(orientation_sets[i]):
            sum += abs(orientation_sets[i][j] - label_sets[i][j]['Theta'])
            num += 1
    return sum/num

def parselabel(label_path):
    label_sets = []
    round = 0
    with open(label_path) as xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for frame in root.iter('frame'):
            if round%8 == 0:
                label_sets.append([])
            # found the exact frame, now parse the boxed and orientation
            for target_list in frame.iter('target_list'):
                    for target in target_list.iter('target'):
                        box_2d = None
                        for box in target.iter('box'):
                            b = box.attrib
                            #print(b)
                            top_left = (int(float(b["left"])), int(float(b["top"])))
                            bottom_right = (top_left[0] + int(float(b["width"])), top_left[1] + int(float(b["height"])))
                            box_2d = [top_left, bottom_right]
                        for attribute in target.iter('attribute'):
                            orientation_GT = float(attribute.attrib["orientation"])/180*np.pi
                            theta = 0
                            if orientation_GT >= 0 and orientation_GT <= 1.5 * np.pi:
                                theta = 1/2 * np.pi - orientation_GT
                            else:
                                theta = 5/2 * np.pi - orientation_GT
                        label = {
                            'Box_2D': box_2d,
                            'Theta': theta
                        }
                        label_sets[round//8].append(label)
    return label_sets