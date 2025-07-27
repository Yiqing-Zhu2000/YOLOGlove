# This is for primary test no need to use 
# this file haven't use functions to change. 
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
from gensim.models import KeyedVectors
import sys
import numpy as np
from src.YOLO_utils import save_YOLO_patches, draw_YOLOboxes_on_Img



# load model
model = YOLO('yolov8x.pt')

# image path
img_path = os.path.join('houses', 'house1.jpg')

# run model
results = model(img_path, imgsz=1024, augment=True)

# open image
img = Image.open(img_path).convert("RGB")
os.makedirs('yoloOutput', exist_ok=True)

# save patches
boxes = results[0].boxes.xyxy.cpu().numpy()
print(f"Number of box detected: {len(boxes)} ")

for idx, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    patch = img.crop((x1, y1, x2, y2))
    patch.save(f'yoloOutput/patch_{idx}.jpg')
    # print(f"patch_{idx}.jpg saved")

# draw boxes
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 80)
except:
    font = ImageFont.load_default()

names = model.names
os.makedirs("output", exist_ok=True)

for box, cls, conf in zip(
    boxes,
    results[0].boxes.cls.cpu().numpy(),
    results[0].boxes.conf.cpu().numpy()
):
    x1, y1, x2, y2 = map(int, box)
    label = f"{names[int(cls)]} {conf:.2f}"
    draw.rectangle([x1, y1, x2, y2], outline="cyan", width=4)
    text_y = y1 - 80 if y1 - 80 > 0 else y1 + 5
    draw.text((x1, text_y - 20), label, fill="cyan", font=font)

output_path = "output/yolo_boxed.jpg"
img.save(output_path)
print(f"✅ Saved the boxed image to {output_path}")

# =============== Use Glove model ==============
def get_glove_vector(label, glove_model):
    try:
        return glove_model[label]
    except KeyError:
        # for many words. 
        words = label.split()
        vectors = []
        for word in words:
            if word in glove_model:
                vectors.append(glove_model[word])
        if vectors:
            return sum(vectors) / len(vectors)
        else:
            return None

# YOLO output labels:
# extract detected index.
YOLOclass_ids = results[0].boxes.cls.cpu().numpy().astype(int)
YOLO_labels = [names[i] for i in YOLOclass_ids]   # change index -> text labels
YOLO_labels_unique = list(set(YOLO_labels))     # unique labels. 
print("labels unique", YOLO_labels_unique)

# change to apply glove model (.word2vec.txt)
glove_path = "glove6B/glove.6B.50d.word2vec.txt"  
model = KeyedVectors.load_word2vec_format(glove_path)

# === target word === vegetable, pumpkin, vehicle
target_word = "vegetable"

# === similarity threshold set, similarity resutls ===
threshold = 0.45
found_vegetables = []

# === for each label, compute similarity ===
target_vec = get_glove_vector(target_word, model)
for label in YOLO_labels_unique:
    label_vec = get_glove_vector(label, model)
    if label_vec is None:
        print(f"'{label}' not in GloVe vocabulary, skipped.")
        continue
    # calculate cosine similarity 
    sim = np.dot(label_vec, target_vec) / (np.linalg.norm(label_vec) * np.linalg.norm(target_vec))
    print(f"{label} vs {target_word} similarity = {sim:.3f}")

    if sim > threshold:
        found_vegetables.append(label)


# === output result ===
print("\n=== Final Judgment ===")
if found_vegetables:
    print(f"✅ image contains the target word. match with: {found_vegetables}")
else:
    print("❌ NO target word object in this image.")
 