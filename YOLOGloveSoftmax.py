import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
from gensim.models import KeyedVectors
import sys
import numpy as np
import torch
from src.YOLO_utils import save_YOLO_patches, draw_YOLOboxes_on_Img,labels_vs_target_similarity_glove
from src.Glove_utils import get_glove_vector

# Paths
COCO_img_dir = "D:/019_2025summer/Datasets/COCOSearch18-images-TP/images/"
glove_path = "glove6B/glove.6B.300d.word2vec.txt"
img_path = os.path.join(COCO_img_dir,"bottle",'000000009527.jpg')
output_dir = "output/"

# Load model, read the thresholds file
os.makedirs(output_dir, exist_ok=True)
YOLO_model = YOLO('yolov8x.pt')
glove_model = KeyedVectors.load_word2vec_format(glove_path)

# ============ Prepare ======================
# ======= target word, similarity threshold === vegetable, pumpkin, vehicle
# if want to test many target words, 1. check whether in 18 categories, 2. set relative thred.
target_word = "bowl"

# ========== Prepare ==========
img = Image.open(img_path).convert("RGB")
names = YOLO_model.names

# ========== RUN YOLO model on ONE image ==============
# verbose=False -> to close the print for YOLO log output infor.                         
YOLOresutls = YOLO_model(img_path, imgsz=1024, augment=True, verbose=False)
boxes = YOLOresutls[0].boxes.xyxy.cpu().numpy()

# =========== get unique YOLO detected labels ================
YOLOclass_ids = YOLOresutls[0].boxes.cls.cpu().numpy().astype(int)
YOLO_labels = [names[i] for i in YOLOclass_ids]   # change index -> text labels
YOLO_labels_unique = list(set(YOLO_labels))     # unique labels. 
print("labels unique", YOLO_labels_unique)



# =============== get similarities of labels vs. target word, For ONE image =========
labels_sims = labels_vs_target_similarity_glove(glove_model, YOLO_labels_unique, target_word)
# ======= output result ======

# softmax over all similarity scores, labels_sims is 1D
similarities_tensor = torch.tensor(labels_sims)
probs = torch.nn.functional.softmax(similarities_tensor, dim=-1).numpy()

# === output results ===
print("\n=== Softmax Similarities ===")
for label, prob in zip(YOLO_labels_unique, probs):
    print(f"{label}: softmax prob = {prob:.3f}")

# best match:
sorted_probs = sorted(zip(YOLO_labels_unique, probs), key=lambda x: x[1], reverse=True)
top_label, top_prob = sorted_probs[0]
second_label, second_prob = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0.0)

# === output result ===
print("=========Our target object is: ",target_word)
if top_prob - second_prob >= 0.1:   # if one much bigger
    print(f"✅ Target likely exists: '{top_label}' (top={top_prob:.3f}, second={second_prob:.3f})")
else:
    print(f"❌ Ambiguous: no strong dominant target (top={top_prob:.3f}, second={second_prob:.3f})")


 