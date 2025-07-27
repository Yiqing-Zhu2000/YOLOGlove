import json
import os
from collections import defaultdict

output_dir = "new_jsonFile/"
os.makedirs("new_jsonFile", exist_ok=True)

def get_deduplicated_task_name_bbox_json(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    seen = set()
    deduplicated = []

    for item in data:
        key = (item["task"], item["name"])
        if key not in seen:
            seen.add(key)
            deduplicated.append({
                "task": item["task"],
                "name": item["name"],
                "bbox": item["bbox"]
            })

    print(f"Deduplicated samples: {len(deduplicated)}")
    with open(output_path, "w") as f:
        json.dump(deduplicated, f, indent=2)
    return deduplicated

# ======== run and generate new json files =================
train_split1 = "D:/019_2025summer/Datasets/COCOSearch18-fixations-TP/coco_search18_fixations_TP_train_split1.json"
new_train_split1= os.path.join(output_dir, "coco18_train_split1_deduplicated_task_name_bbox.json")
get_deduplicated_task_name_bbox_json(train_split1, new_train_split1)

val_split1 = "D:/019_2025summer/Datasets/COCOSearch18-fixations-TP/coco_search18_fixations_TP_validation_split1.json"
new_val_split1= os.path.join(output_dir, "coco18_validation_split1_deduplicated_task_name_bbox.json")
get_deduplicated_task_name_bbox_json(val_split1, new_val_split1)


# ============ check new json file =================
def check_samples_for_eachTask(json_path):
    # json_path: new_train_split1's path
    # read deduplicated file
    with open(json_path, "r") as f:
        deduplicated_data = json.load(f)

    # 统计每个 task 的样本数
    task_counter = defaultdict(int)
    for item in deduplicated_data:
        task_counter[item["task"]] += 1

    # 打印结果
    print(json_path, "[Task-wise sample counts after deduplication]")
    for task, count in sorted(task_counter.items(), key=lambda x: -x[1]):
        print(f"{task:15s}: {count}")

# check
# check_samples_for_eachTask(new_train_split1)
# check_samples_for_eachTask(new_val_split1)

# found that for dataset split1 method: train: 70%, validation: 10%, rest(test): 20%