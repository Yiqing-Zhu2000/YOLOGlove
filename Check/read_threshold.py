import pandas as pd
import os

# Read the CSV file
df = pd.read_csv("output/category_thresholds.csv")

# Display the table
# print(df.head())

# print(df["category"])           # 所有类别名称
# print(df["midpoint_threshold"]) # 所有 midpoint 阈值


threshold_for_bottle = df[df["category"] == "bottle"]["midpoint_threshold"].values[0]
# print("threshold_for_bottle", threshold_for_bottle)

target_word = "vegetable"
if target_word in df["category"].values:
    print("yes, it is in it ")
else:
    print("no target word in COCO 18 categories.")
