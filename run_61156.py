
import os
import time
import copy

import numpy as np
import cv2
import math

import torch
import torch.nn as nn

from torch_lib.Dataset import *
from torch_lib import Model, ClassAverages
from torch.autograd import Variable
from torchvision.models import vgg

from library.Math import *
from library.Plotting import *
from yolo.yolo import cv_Yolo

import argparse
import rospy

#msg
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="sample_data/MVI_40743",
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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def callback(data, args):
    model = args[0] 
    yolo = args[1]
    averages = args[2]
    angle_bins = args[3]

    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    yolo_img = np.copy(img)
    truth_img = np.copy(img)
    detections = yolo.detect(yolo_img)

    orientations = []

    for detection in detections:

        if not averages.recognized_class(detection.detected_class):
            continue
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

        input_tensor = torch.zeros([1,3,224,224]).cuda()
        input_tensor[0,:,:,:] = input_img

        [orient, conf, dim] = model(input_tensor)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]

        dim += averages.get_item(detected_class)

        argmax = np.argmax(conf)
        print(conf[argmax])
        if conf[argmax] < 4.5:
            continue
        orient = orient[argmax, :]
        cos = orient[0]
        sin = orient[1]
        alpha = np.arctan2(sin, cos)
        alpha += angle_bins[argmax]
        alpha -= np.pi

            #print("theta_L", alpha)
            #print("theta_ray", theta_ray)
            #print("2D box", box_2d)
        location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)

        if not FLAGS.hide_debug:
            print('Estimated pose: %s'%location)
            print('Estimated orient: %s'%orient)

    numpy_vertical = np.concatenate((truth_img, img), axis=0)
    cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)

    cv2.waitKey(1)
        
def listener(args):
	rospy.init_node('ThreeDBBOX' ,anonymous = True)
	rospy.Subscriber('usb_cam/image_rect_color/raw',Image,callback, callback_args=args)
	rospy.spin()

def main():
    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        # torch.cuda.set_device(1)
        # device = torch.device('cuda:0')
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1] )#map_location=device
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    args = (model, yolo, averages, angle_bins)

    listener(args)

if __name__ == '__main__':

    main()
