"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""


from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    union = detected_x * detected_y + label_x * label_y
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

def parse_label(frame_num):
    label_path = "DETRAC-Test-Annotations-XML/MVI_39371.xml"
    labels = []
    with open(label_path) as xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for frame in root.iter('frame'):
            if int(frame.attrib["num"]) == frame_num:
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
                        labels.append(label)
                break
    return labels

parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        device = torch.device('cpu')
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2)#.cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir
    if FLAGS.video:
        if FLAGS.image_dir == "eval/image_2/" and FLAGS.cal_dir == "camera_cal/":
            image_dir = "eval/video/2011_09_26/image_2/"
            cal_dir = "eval/video/2011_09_26/"


    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"

    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
        #print(ids)
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for img_id in ids:
        if len(img_id) == 0:
            continue
        start_time = time.time()

        img_file = img_path + img_id + ".png"
        #print(img_file)
        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        #print(img.shape)
        yolo_img = np.copy(truth_img)

        detections = yolo.detect(yolo_img)

        orientations = []

        for detection in detections:

            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            #print(proj_matrix)
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1,3,224,224])#.cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            #print("theta_L", alpha)
            #print("theta_ray", theta_ray)
            #print("2D box", box_2d)
            if FLAGS.show_yolo:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

            if not FLAGS.hide_debug:
                print('Estimated pose: %s'%location)
                print('Estimated orient: %s'%orient)

        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        else:
            cv2.imshow('3D detections', img)

        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')

        if FLAGS.video:
            cv2.waitKey(1)
        else:
            cv2.waitKey(1)
            #if cv2.waitKey(0) != 32: # space bar
            #   exit()

if __name__ == '__main__':
    main()
