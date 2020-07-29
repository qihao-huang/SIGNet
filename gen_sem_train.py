import os
import cv2
import glob
from shutil import copyfile

# Tensorflow
import tensorflow as tf
print(tf.__version__)

# I/O libraries
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

# Helper libraries
import matplotlib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv
from tqdm import tqdm
import IPython
from sklearn.metrics import confusion_matrix
from tabulate import tabulate


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, pretrain_model_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = tf.GraphDef.FromString(open(pretrain_model_path,'rb').read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image, INPUT_TENSOR_NAME = 'ImageTensor:0', OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.
            INPUT_TENSOR_NAME: The name of input tensor, default to ImageTensor.
            OUTPUT_TENSOR_NAME: The name of output tensor, default to SemanticPredictions.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        # NOTE: use original image to predict
        batch_seg_map = self.sess.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: [np.asarray(image)]})
        seg_map = batch_seg_map[0]  # expected batch size = 1
        if len(seg_map.shape) == 2:
            seg_map = np.expand_dims(seg_map,-1)  # need an extra dimension for cv.resize
        seg_map = cv.resize(seg_map, (width,height), interpolation=cv.INTER_NEAREST)
        
        return seg_map


if __name__ == "__main__":
    pretrain_model_path = "/userhome/34/h3567721/projects/Depth/deeplab/deeplabv3_cityscapes_train/frozen_inference_graph.pb"
    MODEL = DeepLabModel(pretrain_model_path)

    # /userhome/34/h3567721/dataset/kitti/kitti_raw_eigen/2011_09_26_drive_0001_sync_02/***.jpg

    base_dir = "/userhome/34/h3567721/dataset/kitti"
    kitti_raw_eigen_dir = os.path.join(base_dir, "kitti_raw_eigen")
    kitti_raw_eigen_sem_dir = os.path.join(base_dir, "kitti_raw_eigen_sem")
    make_dir(kitti_raw_eigen_sem_dir)

    all_dirs = [o for o in os.listdir(kitti_raw_eigen_dir) if os.path.isdir(os.path.join(kitti_raw_eigen_dir,o))]

    for img_dir in all_dirs:
        print("processing ", img_dir)
        img_path_list = glob.glob(os.path.join(kitti_raw_eigen_dir, img_dir) + "/*.jpg")

        make_dir(os.path.join(kitti_raw_eigen_sem_dir, img_dir))

        for i in range(len(img_path_list)):
            src_img_path = img_path_list[i]
            rgb_img = Image.open(src_img_path)
            
            rgb_img_1 = rgb_img.crop((0, 0, 416, 128))
            rgb_img_2 = rgb_img.crop((416, 0, 832, 128)) 
            rgb_img_3 = rgb_img.crop((832, 0, 1248, 128)) 

            seg_map_1 = MODEL.run(rgb_img_1)
            seg_map_2 = MODEL.run(rgb_img_2)
            seg_map_3 = MODEL.run(rgb_img_3)

            pack_img = np.hstack([seg_map_1, seg_map_2, seg_map_3])

            dst_img_path = src_img_path.replace("kitti_raw_eigen", "kitti_raw_eigen_sem")
            dst_img_path = dst_img_path.replace("jpg", "npy")
            np.save(dst_img_path, pack_img)

# 20
# LABEL_NAMES = np.asarray([
#     'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
#     'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
#     'bus', 'train', 'motorcycle', 'bicycle', 'void']) 