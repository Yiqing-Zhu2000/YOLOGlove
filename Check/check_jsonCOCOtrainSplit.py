# check COCO train split 1,2 json files. 
import os

def list_subfolders(parent_dir):
    """
    List all subfolder names under a given parent directory.

    Args:
        parent_dir (str): Path to the main folder.

    Returns:
        List[str]: List of subfolder names (not full paths).
    """
    subfolders = [
        name for name in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, name))
    ]
    return subfolders

# # 示例使用
# if __name__ == "__main__":
#     COCO_img_dir = "D:/019_2025summer/Datasets/COCOSearch18-images-TP/images/"
#     subfolder_names = list_subfolders(COCO_img_dir)
#     print("Found subfolders:", subfolder_names)
#     print("lenght", len(subfolder_names))
    

COCO_train1_dir = "D:/019_2025summer/Datasets/COCOSearch18-fixations-TP/coco_search18_fixations_TP_train_split1.json"
COCO_train2_dir = "D:/019_2025summer/Datasets/COCOSearch18-fixations-TP/coco_search18_fixations_TP_train_split2.json"

# # split1 files have the same train/valid split as was done in the paper.
# import json
# from collections import Counter

# with open(COCO_train2_dir, "r") as f:
#     data = json.load(f)

# task_counts = Counter()
# for item in data:
#     task = item.get("task", None)
#     if task:
#         task_counts[task] += 1

# # 打印排序后的结果
# for category, count in task_counts.most_common():
#     print(f"{category}: {count}")

############################################################################3
import json
from collections import defaultdict

# # 加载 JSON 数据
# with open(COCO_train1_dir, "r") as f:
#     data = json.load(f)

# # 记录每个 (name, task) 对应的 bbox
# bbox_dict = defaultdict(set)

# for item in data:
#     name = item.get("name")
#     task = item.get("task")
#     bbox = tuple(item.get("bbox", []))  # 转为不可变类型用于 set 去重
#     key = (name, task)
#     bbox_dict[key].add(bbox)

# # 找出 name+task 相同但 bbox 不一致的项
# inconsistent = {key: list(bboxes) for key, bboxes in bbox_dict.items() if len(bboxes) > 1}

# print(f"Total (name, task) pairs with inconsistent bbox: {len(inconsistent)}")
# for (name, task), bboxes in list(inconsistent.items())[:10]:  # 只打印前10个
#     print(f"{name} + {task}: {bboxes}")


#####################################################
import json
from collections import defaultdict

# with open(COCO_train1_dir, "r") as f:
#     data = json.load(f)

# grouped = defaultdict(list)

# # 把相同 name+task+bbox 的样本放在一起
# for item in data:
#     name = item["name"]
#     task = item["task"]
#     bbox = tuple(item["bbox"])
#     key = (name, task, bbox)
#     grouped[key].append(item)

# # 检查同组样本是否还有其他字段不同
# for key, group in grouped.items():
#     if len(group) > 1:
#         # 把所有样本转成字符串做差异比较
#         json_strs = [json.dumps(d, sort_keys=True) for d in group]
#         if len(set(json_strs)) > 1:
#             print(f"\n{name} + {task} + {bbox} has {len(group)} entries with other differences.")
#             for d in group:
#                 print(f"  subject: {d['subject']}, correct: {d['correct']}, RT: {d['RT']}")


######################################################################33

with open(COCO_train1_dir, "r") as f:
    data = json.load(f)

task_sequence = []
seen = set()

for item in data:
    task = item["task"]
    if task != task_sequence[-1] if task_sequence else None:
        task_sequence.append(task)

# 检查是否有重复 task 出现在后面
task_first_seen = set()
reordered = False

for t in task_sequence:
    if t in task_first_seen:
        reordered = True
        print(f"⚠️ Task '{t}' appears again after being interrupted.")
    task_first_seen.add(t)

print("\n✅ Task appearance order:")
print(task_sequence)

if not reordered:
    print("\n✅ Good news: All samples for each task appear contiguously (grouped).")
else:
    print("\n❌ Some task samples are interleaved — not grouped together.")
