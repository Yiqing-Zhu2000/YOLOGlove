import torch
from ultralytics import YOLO
from gensim.models import KeyedVectors
import sys
import os

# simulate the YOLO lables output, and compare with target "vegetable":
# Word2Vec file 
glove_path = "glove6B/glove.6B.300d.word2vec.txt"  
model = KeyedVectors.load_word2vec_format(glove_path)
print("glove path used:", glove_path)

# === simulate YOLO output labels list ===
# labels = ["cabbage", "dog", "car", "broccoli", "chair", "apple"]
labels = [ "dog", "car", "cat", "chair", "fruit"]


# === target word === vegetable
target_word = "cat"

# === similarity threshold set, similarity resutls ===
threshold = 0.5
found_vegetables = []

# === for each label, compute similarity ===
for label in labels:
    try:
        sim = model.similarity(label, target_word)
        print(f"{label} vs {target_word} similarity = {sim:.3f}")
        if sim > threshold:
            found_vegetables.append(label)
    except KeyError:
        print(f"'{label}' not in GloVe vocabulary, skipped.")

# === output result ===
print("\n=== Final Judgment ===")
if found_vegetables:
    print(f"✅ image contains the target word: {found_vegetables}")
else:
    print("❌ NO target word object in this image.")
 