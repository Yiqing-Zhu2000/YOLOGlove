import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

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
    font = ImageFont.truetype("arial.ttf", 50)
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
    draw.text((x1, y1 - 20), label, fill="cyan", font=font)

output_path = "output/v8x_boxed.jpg"
img.save(output_path)
print(f"✅ Saved the boxed image to {output_path}")
