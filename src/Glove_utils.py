import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
from gensim.models import KeyedVectors
import sys
import numpy as np
import torch

# =============== Use Glove model ==============
def get_glove_vector(label, glove_model):
    try:
        # work for one word's vector 
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
        
