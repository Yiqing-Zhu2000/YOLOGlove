import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
from gensim.models import KeyedVectors
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.Glove_utils import get_glove_vector
from collections import defaultdict


def save_YOLO_patches(img, boxes):
    """
    img: the opened original img
    boxes: boxes from YOLO results[0].boxes.xyxy.cpu().numpy()
    """
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        patch = img.crop((x1, y1, x2, y2))
        os.makedirs('yoloOutput', exist_ok=True)
        patch.save(f'yoloOutput/patch_{idx}.jpg')
        # print(f"patch_{idx}.jpg saved")
    return 

# draw boxes
# ======== draw boxes on the original img and store new img to "output/yolo_boxed.jpg" ======
def draw_YOLOboxes_on_Img(img_path, boxes, mod_names, YOLOresutls):
    """
    boxes: can be square boxes modified based on detected boxes
    mod_names: use model.names
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, cls, conf in zip(
        boxes,
        YOLOresutls[0].boxes.cls.cpu().numpy(),
        YOLOresutls[0].boxes.conf.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        label = f"{mod_names[int(cls)]} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=4)
        draw.text((x1, y1 - 20), label, fill="cyan", font=font)
    
    os.makedirs("output", exist_ok=True)
    output_path = "output/yolo_boxed.jpg"
    img.save(output_path)
    print(f"Saved the boxed image to {output_path}")
    return 


# ============ functions used in set_threshold =========
def group_by_task(deduplicated_data):
    """
    based on task to classify, and under each task, 
    store in form:  'task': [{'name': 'name1.jpg', 'bbox': [x, y, w, h]}, 
                             {'name': 'name2.jpg', 'bbox': [x2, y2, w2, h2]},...]
    and return this 

    Args:
        deduplicated_data (List[Dict]): [{task, name, bbox}, ...]
        read from new json file
    """
    grouped = defaultdict(list)

    for item in deduplicated_data:
        grouped[item["task"]].append({
            "name": item["name"],
            "bbox": item["bbox"]
        })

    return dict(grouped)


# ============ yolo detects for one image =========
def YOLO_detect_labels_boxes(img_path, image_name, YOLO_model):
    # run YOLO model
    # verbose=False -> to close the print for YOLO log output infor.  
    results = YOLO_model(img_path, imgsz=1024, augment=True, verbose=False) 
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    pred_labels = [YOLO_model.names[int(i)] for i in results[0].boxes.cls.cpu().numpy()]   # match with pred_boxes
    return pred_boxes, pred_labels

# =============== compute IOU ==================
def compute_iou(boxA, boxB):
    # compute the intersection or
    # boxA: yolo box [x1, y1, x2, y2]
    # boxB: COCO bbox [x, y, w, h] target label's bbox (ground truth). 
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ============= for ONE of training images ============
def get_OneSingal_N_noise_sim(target_vec, bbox, pred_boxes,pred_labels,glove_model):
    """
    target_vec: glove vec computed for the task label
    bbox: the bbox for the target object
    pred_boxes: yolo boxes [x1, y1, x2, y2]
    pred_labels: yolo detected labels corresponding to yolo boxes
    glove_model: the loaded glove model 

    return: one signal sim, the list of noise sims.
    """
    signal_sim = 0.0
    l_noise_sims = []

    # Get the Most matched box index
    matched_box_idx = -1
    max_iou = 0
    for i, (pred_box, label) in enumerate(zip(pred_boxes, pred_labels)):
        iou = compute_iou(pred_box, bbox)
        # and iou >= 0.5   # check whether YOLO pred_box and target bbox have at least 50% overlap
        if iou > max_iou:
            matched_box_idx = i
            max_iou = iou

    if matched_box_idx != -1:   # find the most matched yolo box + label
        signal_label = pred_labels[matched_box_idx]
        label_vec = get_glove_vector(signal_label, glove_model)
        signal_sim = np.dot(label_vec, target_vec) / (np.linalg.norm(label_vec) * np.linalg.norm(target_vec))
       
        # removing duplicates using set(), and then keeping only those labels that are not equal to the signal_label
        uni_noisy_labels = [l for l in set(pred_labels) if l != signal_label]
        for l in uni_noisy_labels:
            l_vec = get_glove_vector(l, glove_model)
            noisy_sim = np.dot(l_vec, target_vec) / (np.linalg.norm(l_vec) * np.linalg.norm(target_vec))
            l_noise_sims.append(noisy_sim)

    return signal_sim, l_noise_sims

# ================ save distribution plot =============
def plot_signal_vs_noise(signal_distri, noise_distri, threshold, save_path=None, filename='signal_vs_noise.png'):
    """
    Plots histogram comparing signal and noise cosine similarity distributions, and optionally saves the plot.

    Parameters:
    - signal_distri (list or array): Cosine similarities for signal.
    - noise_distri (list or array): Cosine similarities for noise.
    - threshold (float): Decision threshold to be shown as a vertical line.
    - save_path (str or None): Directory to save the plot. If None, the plot won't be saved. eg. "output/"
    - filename (str): Name of the file to save the plot as (default: 'signal_vs_noise.png').
    """
    plt.figure(figsize=(8, 6))
    plt.hist(noise_distri, bins=30, alpha=0.5, label='Noise', color='red')
    plt.hist(signal_distri, bins=30, alpha=0.5, label='Signal', color='green')
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    plt.legend()
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Signal vs Noise Distribution")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
        print(f"Plot saved to {full_path}")

    # plt.show()
    return 

# =============== get similarities of labels vs. target word, For ONE image =========
def labels_vs_target_similarity_glove(glove_model, YOLO_labels_unique, target_word):
    """
    This is for one image
    Args:
        glove_model: A pretrained GloVe model loaded with KeyedVectors.
        YOLO_labels_unique: A list of unique predicted labels from YOLO for ONE image
        target_word: The target category word to compare against.
    Return:
        found_overThred: A list of predicted labels whose cosine similarity with the target word.
    """
    # === for each label, compute similarity ===
    labels_sims = []
    target_vec = get_glove_vector(target_word, glove_model)
    for label in YOLO_labels_unique:
        label_vec = get_glove_vector(label, glove_model)
        if label_vec is None:
            print(f"'{label}' not in GloVe vocabulary, skipped.")
            continue
        # calculate cosine similarity 
        sim = np.dot(label_vec, target_vec) / (np.linalg.norm(label_vec) * np.linalg.norm(target_vec))
        print(f"{label} vs {target_word} similarity = {sim:.3f}")

        labels_sims.append(sim)
    return labels_sims

