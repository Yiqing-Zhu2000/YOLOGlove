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
import pandas as pd
import csv

print("Start to run Set_18thresholds.py")

# Paths
COCO_img_dir = "images/"
json_path = "new_jsonFile/coco18_train_split1_deduplicated_task_name_bbox.json"
glove_path = "glove6B/glove.6B.300d.word2vec.txt"
output_dir = "output/"

# Prepare, load model, read files
os.makedirs(output_dir, exist_ok=True)
YOLO_model = YOLO('yolov8x.pt')
glove_model = KeyedVectors.load_word2vec_format(glove_path)
with open(json_path, "r") as f:
    deduplicated_data = json.load(f)

# Group items by task
task_to_items = group_by_task(deduplicated_data)
# print(task_to_items)

# Process each task
thresholds_result = {}
for task, items in task_to_items.items():

    # items = items[:3]  # ← only use the first 3 samples of each category JUST for local test
    
    # for this task, collect singal vs noise sims.
    signal_sims, noise_sims = [], []
    print(f"\nProcessing category: {task} ({len(items)} samples)")
    # run glove to get vec. of traget task words
    target_vec = get_glove_vector(task, glove_model)

    for item in items:
        img_name = item["name"]
        bbox = item["bbox"]
        img_path = os.path.join(COCO_img_dir, task, img_name)

        # run yolo for each img
        try:
            pred_boxes, pred_labels = YOLO_detect_labels_boxes(img_path, "noneed", YOLO_model)
        except Exception as e:
            print(f"Error on image {img_name}: {e}")
            continue

        # for each img for this task, based on yolo pred_boxes to match with ground truth bbox for img
        # get the most matching yolo box by iou, its relative label is the signal label
        # other rest yolo pred labels are noise label, all compare with target vec. Store cosine similarities.
        signal_sim, list_noise_sims = get_OneSingal_N_noise_sim(target_vec, bbox, pred_boxes,pred_labels,glove_model)
        # combine one img info. to collection for this task
        signal_sims.append(signal_sim)
        noise_sims.extend(list_noise_sims)

    if not signal_sims or not noise_sims:
        print(f"Skipping category {task} due to insufficient data.")
        continue

    signal_distri = np.array(signal_sims)
    noise_distri = np.array(noise_sims)
    mu_signal = np.mean(signal_distri)
    mu_noise = np.mean(noise_distri)
    midpoint_threshold = (mu_signal + mu_noise) / 2

    # EER Threshold
    y_true = np.concatenate([np.ones_like(signal_distri), np.zeros_like(noise_distri)])
    y_scores = np.concatenate([signal_distri, noise_distri])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds[eer_idx]

    thresholds_result[task] = {
        "midpoint_threshold": float(midpoint_threshold),
        "eer_threshold": float(eer_threshold),
        "signal_size": len(signal_distri),
        "noise_size": len(noise_distri)
    }

    # Plot and save figure
    plot_path = os.path.join(output_dir,"signal_vs_noise_thres")
    plot_signal_vs_noise(signal_distri, noise_distri, midpoint_threshold, 
                         save_path=plot_path, filename = f"{task}_threshold_plot.png")

# Save thresholds to CSV file
csv_path = os.path.join(output_dir, "category_thresholds.csv")
# Convert the thresholds dictionary to a DataFrame
df = pd.DataFrame.from_dict(thresholds_result, orient="index")
df.index.name = "category"  # set the row index name
df.reset_index(inplace=True)  # move category back to a column

# Save as CSV with rounded float precision
df.to_csv(csv_path, index=False)   # float_format="%.4f"

print("\nThreshold computation complete. Results saved to:", csv_path)