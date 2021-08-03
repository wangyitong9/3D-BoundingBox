import numpy as np
import math
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
