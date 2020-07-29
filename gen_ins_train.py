import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger

import os
import numpy as np
import cv2
import random
import glob
from PIL import Image

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == "__main__":
    # /userhome/34/h3567721/dataset/kitti/kitti_raw_eigen/2011_09_26_drive_0001_sync_02/***.jpg

    base_dir = "/userhome/34/h3567721/dataset/kitti"
    kitti_raw_eigen_dir = os.path.join(base_dir, "kitti_raw_eigen")
    kitti_raw_eigen_ins_dir = os.path.join(base_dir, "kitti_raw_eigen_ins")
    make_dir(kitti_raw_eigen_ins_dir)

    all_dirs = [o for o in os.listdir(kitti_raw_eigen_dir) if os.path.isdir(os.path.join(kitti_raw_eigen_dir,o))]

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    for img_dir in all_dirs:
        print("processing ", img_dir)
        img_path_list = glob.glob(os.path.join(kitti_raw_eigen_dir, img_dir) + "/*.jpg")

        make_dir(os.path.join(kitti_raw_eigen_ins_dir, img_dir))

        for i in range(len(img_path_list)):
            src_img_path = img_path_list[i]
            rgb_img = Image.open(src_img_path).convert('RGB')
 
            # crop and convert RGB to BGR
            rgb_img_1 = np.array(rgb_img.crop((0, 0, 416, 128)))
            rgb_img_1 = rgb_img_1[:, :, ::-1].copy() 

            rgb_img_2 = np.array(rgb_img.crop((416, 0, 832, 128)))
            rgb_img_2 = rgb_img_2[:, :, ::-1].copy() 

            rgb_img_3 = np.array(rgb_img.crop((832, 0, 1248, 128)))
            rgb_img_3 = rgb_img_3[:, :, ::-1].copy() 

            output_1 = predictor(rgb_img_1)
            output_2 = predictor(rgb_img_2)
            output_3 = predictor(rgb_img_3)

            mask_1 = output_1['instances'].pred_masks.cpu().numpy().transpose([1,2,0])
            mask_2 = output_2['instances'].pred_masks.cpu().numpy().transpose([1,2,0])
            mask_3 = output_3['instances'].pred_masks.cpu().numpy().transpose([1,2,0])

            ins_class_1 = output_1['instances'].pred_classes.cpu().numpy()
            ins_1_0 = np.zeros((mask_1.shape[0],mask_1.shape[1]), dtype=int)
            ins_1_1 = np.zeros((mask_1.shape[0],mask_1.shape[1]), dtype=int)

            ins_class_2 = output_2['instances'].pred_classes.cpu().numpy()
            ins_2_0 = np.zeros((mask_2.shape[0],mask_2.shape[1]), dtype=int)
            ins_2_1 = np.zeros((mask_2.shape[0],mask_2.shape[1]), dtype=int)

            ins_class_3 = output_3['instances'].pred_classes.cpu().numpy()
            ins_3_0 = np.zeros((mask_3.shape[0],mask_3.shape[1]), dtype=int)
            ins_3_1 = np.zeros((mask_3.shape[0],mask_3.shape[1]), dtype=int)

            for i, sig_class in enumerate(ins_class_1):
                ins_1_0[mask_1[:,:,i]] = sig_class+1

            for i, sig_class in enumerate(ins_class_1):
                ins_1_1[mask_1[:,:,i]] = i+1

            for i, sig_class in enumerate(ins_class_2):
                ins_2_0[mask_2[:,:,i]] = sig_class+1

            for i, sig_class in enumerate(ins_class_2):
                ins_2_1[mask_2[:,:,i]] = i+1

            for i, sig_class in enumerate(ins_class_3):
                ins_3_0[mask_3[:,:,i]] = sig_class+1

            for i, sig_class in enumerate(ins_class_3):
                ins_3_1[mask_3[:,:,i]] = i+1

            ins_pack_0 = np.hstack([ins_1_0, ins_2_0, ins_3_0]) # (128, 1248)
            ins_pack_1 = np.hstack([ins_1_1, ins_2_1, ins_3_1]) # (128, 1248)
             
            ins_pack_0 = np.expand_dims(ins_pack_0,axis=2) # (128, 1248, 1)
            ins_pack_1 = np.expand_dims(ins_pack_1,axis=2) # (128, 1248, 1)

            ins_cat = np.concatenate((ins_pack_0, ins_pack_1), axis=2) # (128, 1248, 2)

            dst_img_path = src_img_path.replace("kitti_raw_eigen", "kitti_raw_eigen_ins")
            dst_img_path = dst_img_path.replace("jpg", "npy")
            print("save: ", dst_img_path)
            np.save(dst_img_path, ins_cat)
