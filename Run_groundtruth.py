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
from yolo.yolo import Detection

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse
import xml.etree.ElementTree as ET

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None, orient_groundtruth=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    # print("plot orientation", orient_groundtruth)

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d, orient_groundtruth)

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

    filename = "MVI_40152.xml"
    xml_file = "../Annotation/DETRAC-Train-Annotations-XML/" + filename
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    for img_id in ids:
        #print(img_id)
        if len(img_id) == 0:
            continue
        start_time = time.time()

        img_file = img_path + img_id + ".png"
        # print(img_file)
        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        # print(img.shape)

        #yolo_img = np.copy(truth_img)
        #
        #detections = yolo.detect(yolo_img)

        # get detection for this particular frame

        detections = []
        orientation_groundtruth = []

        frame_num = int(img_id[3:])
        #print(frame_num)
        for frame in root.iter('frame'):
            if int(frame.attrib["num"]) == frame_num:
                # found the exact frame, now parse the boxed and orientation

                for target_list in frame.iter('target_list'):
                    for target in target_list.iter('target'):
                        # this is one car
                        box_2d = None
                        class_ = None
                        for box in target.iter('box'):
                            b = box.attrib
                            #print(b)
                            top_left = (int(float(b["left"])), int(float(b["top"])))
                            bottom_right = (top_left[0] + int(float(b["width"])), top_left[1] + int(float(b["height"])))
                            box_2d = [top_left, bottom_right]
                            

                        for attribute in target.iter('attribute'):
                            orientation_groundtruth.append(float(attribute.attrib["orientation"])/180*np.pi)
                            class_ = attribute.attrib["vehicle_type"]
                            detections.append(Detection(box_2d, class_))

                break                 


        for index, detection in enumerate(detections):

            if not averages.recognized_class(detection.detected_class):
                # print("here")
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                #print("try detected")
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                # print("throw")
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            # print(proj_matrix)
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1,3,224,224])#.cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            # print("ggggg", averages.get_item(detected_class))
            dim += averages.get_item(detected_class)

            """
            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi
            """
            theta = 0
            if orientation_groundtruth[index] >= 0 and orientation_groundtruth[index] <= 1.5 * np.pi:
                theta = 1/2 * np.pi - orientation_groundtruth[index]
            else:
                theta = 5/2 * np.pi - orientation_groundtruth[index]
            alpha = theta - theta_ray

            if FLAGS.show_yolo:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img, orientation_groundtruth[index])
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
            #     exit()

if __name__ == '__main__':
    main()
