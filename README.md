# YOLOGlove
Based on COCO Search18 datasets to do category search. 
* [YOLO (ultralytics) github](https://github.com/ultralytics/ultralytics): I used `YOLOv8x` version here to detect small objects and labels. (It's a stable version.) It can be changed to use the latest `YOLOv11` models, which deliver state-of-the-art (SOTA) performance across multiple tasks, including object detection, segmentation, pose estimation, tracking, and classification, leveraging capabilities across diverse AI applications and domains.

* [GloVe github](https://github.com/stanfordnlp/GloVe): Use the pre-trained word vectors,`glove6B\glove.6B.300d.txt` 

## Prepare
1. `src\GloveWord2Vec.py` is used to convert the glove `glove6B\glove.6B.300d.txt` file to `glove6B\glove.6B.300d.word2vec.txt` file, which is the glove pre-trained model we need to use to compute the word vector. For how to use it, I used the Gensim python library, and refered to https://radimrehurek.com/gensim/scripts/glove2word2vec.html and https://radimrehurek.com/gensim/models/keyedvectors.html 

2. The script `src\get_new_json.py` generates cleaned JSON files from the original COCO-Search18 fixation datasets by removing duplicate entries that share the same task and name fields. For each unique `(task, name)` pair, it retains the corresponding `bbox` and saves the result in a new JSON file under the `new_jsonFile/` directory. The output files (e.g., `coco18_train_split1_deduplicated_task_name_bbox.json` and `coco18_validation_split1_deduplicated_task_name_bbox.json`) contain a simplified structure with only the `task`, `name`, and `bbox` fields, ensuring that each task-image combination appears only once. The original COCO-Search18 fixation `*split1` files have the same train(70% image data)/valid(10% image data) split as was done in the paper.

3. Some necessary funcitons are store in `src\GloveWord2Vec.py` and `src\YOLO_utils.py`

4. Check that there are 
`glove6B/glove.6B.300d.word2vec.txt`, 
`yolov8x.py`,
`new_jsonFile/coco18_train_split1_deduplicated_task_name_bbox.json`, 
`COCOSearch18-images-TP.zip` or its unzip folder `images`

## Set the thresholds of cosine similarity for 18 COCO categories based on training data.

- Use ComputeCanada to get the similarity thresholds and store them into an `output\category_thresholds.csv` file. 
- prepare for submit to ComputeCanada.  
1. zip the necessary data, models, and codes in this way
```
zip -r Tmp.zip glove6B new_jsonFile yolov8x.pt COCOSearch18-images-TP.zip src requirements.txt Set_18thresholds.py
```
2. For the `set_18thresholds.sh` file, I run my `Set_18thresholds.py`, and store the output folder back. 
```
chmod 600 set_18thresholds.sh
sbatch set_18thresholds.sh
```
3. After using the training image datasets to get suitable threshold for each category, use a new image as input to check whether it can detect whether the target is existing in this image. 18 thresholds results from training data by using CC stored in `output/category_thresholds.csv`. If there is some upading in methods, remember to upload new thresulds. 

## Use the threshold to check whether input image contain the target category. 
- Use ComputeCanada to run `YOLOGlove_SDT.py` check file. 
1. zip the necessary data, models, and codes in this way
```
zip -r Tmp.zip glove6B yolov8x.pt COCOSearch18-images-TP.zip src output requirements.txt YOLOGlove_SDT.py
```

2. For the `YOLOGlove_SDT.sh` file, I run my `YOLOGlove_SDT.py`, print the results. 
```
chmod 600 YOLOGlove_SDT.sh
sbatch YOLOGlove_SDT.sh
```

3. `YOLOGlove_SDT.py` This is used to testing on one new image, estimating whether there is the target category in this image. 
