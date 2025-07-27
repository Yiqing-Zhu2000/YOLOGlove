# use the signal detection theory — (dprime, maybe use this) to set the threshold for the simialrity. 
# I need to get two distribution. 

import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
from gensim.models import KeyedVectors
import sys
import numpy as np
import json
from src.YOLO_utils import *
from src.Glove_utils import get_glove_vector
# Equal Error Rate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# load YOLO model 
YOLO_model = YOLO('yolov8x.pt')

# Load Glove model (must .word2vec.txt)
glove_path = "glove6B/glove.6B.300d.word2vec.txt"  
glove_model = KeyedVectors.load_word2vec_format(glove_path)

output_dir = "output/"

# one image
# img_path = os.path.join('houses', 'house1.jpg')
COCO_img_dir = "D:/019_2025summer/Datasets/COCOSearch18-images-TP/images/"
# img_path = os.path.join(COCO_img_dir,"bottle",'000000142970.jpg')
# pred_boxes, pred_labels = YOLO_detect_labels_boxes(img_path, "yes", YOLO_model)


# the head 10 samples for COCO bottle images. (Try 10 samples for "bottle" threshold )
# later would need to check there duplicated in "names"
num_samples = 10
with open("D:/019_2025summer/Datasets/COCOSearch18-fixations-TP/coco_search18_fixations_TP_train_split1.json", "r") as f:
        data = json.load(f)
        

signal_sims = []  # for signal distribution. 
noisy_sims = []   # for noise distribution. 
for item in data[:num_samples]:
    img_name = item["name"]
    bbox = item["bbox"]
    target = item["task"]   

    # One of images' path:
    img_path = os.path.join(COCO_img_dir,target,img_name)
    pred_boxes, pred_labels = YOLO_detect_labels_boxes(img_path, "yes", YOLO_model)

    # check and get the most matched box
    matched_box_idx = -1
    max_iou = 0

    for i, (pred_box, label) in enumerate(zip(pred_boxes, pred_labels)):
        iou = compute_iou(pred_box, bbox)
        # and iou >= 0.5
        if iou > max_iou: # this is make sure: detected patch and target patch have at least 50% overlap
            matched_box_idx = i
            max_iou = iou

    # Use the most matched box's label to compute GloVe similarity with target label
    target_vec = get_glove_vector(target,glove_model)
    signal_label = pred_labels[matched_box_idx]
    if matched_box_idx != -1:
        label_vec = get_glove_vector(signal_label, glove_model)
        # calculate cosine similarity 
        sim = np.dot(label_vec, target_vec) / (np.linalg.norm(label_vec) * np.linalg.norm(target_vec))
        signal_sims.append(sim)
        print("THE singal_label",signal_label,"cosine similarity:", sim, "max iou:", max_iou)
    else:
        print("No suitable detected YOLO box with target box!! Max iou", max_iou, "matched_box label", 
              signal_label)

    # get cosine sim with other detected labels to form noisy distribution  
    uni_noisy_labels = [label for label in set(pred_labels) if label != signal_label]
    for l in uni_noisy_labels:
        l_vec = get_glove_vector(l, glove_model)
        sim = np.dot(l_vec, target_vec) / (np.linalg.norm(l_vec) * np.linalg.norm(target_vec))
        noisy_sims.append(sim)


# ========= compute the threshold ===========
# now for these 10 images, I got the signal distribution and the noisy distribution. 
# based on this, to set the threshold. 
signal_distri = np.array(signal_sims)  # cosine similarity of correct boxes
noise_distri = np.array(noisy_sims)    # cosine similarity of incorrect boxes

mu_signal = np.mean(signal_distri)
mu_noise = np.mean(noise_distri)
threshold = (mu_signal + mu_noise) / 2
print("based on the d-prime method, threshold: ", threshold)

# Equal Error Rate
y_true = np.concatenate([np.ones_like(signal_distri), np.zeros_like(noise_distri)])
y_scores = np.concatenate([signal_distri, noise_distri])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 找到 FPR 和 FNR 最接近的点
fnr = 1 - tpr
eer_idx = np.nanargmin(np.absolute(fnr - fpr))
eer_threshold = thresholds[eer_idx]

# plot 
plot_signal_vs_noise(signal_distri, noise_distri, threshold, save_path = output_dir)






