import os
from pathlib import Path
from tqdm import tqdm
import cv2
import json
import numpy as np


def prepare_data(
    file_path,
    root_path="./raw_data/AIC21_Track5_NL_Retrieval/",
    save_path="./extracted_data/",
):
    with open(file_path, "r") as f_r:
        tracks = json.load(f_r)
    track_ids = list(tracks.keys())
    newsize = (256, 256)
    track_ids_v2 = {}

    for track_id in tqdm(track_ids):
        image_sample = tracks[track_id]
        #     print(track_id)
        if image_sample["boxes"][0][-1] > image_sample["boxes"][-1][-1]:
            img = cv2.imread(root_path + image_sample["frames"][0])
            box = image_sample["boxes"][0]
            croped_image = np.array(img)[
                box[1]: box[1] + box[3], box[0]: box[0] + box[2], :
            ]
        else:
            img = cv2.imread(root_path + image_sample["frames"][-1])
            box = image_sample["boxes"][-1]
            croped_image = np.array(img)[
                box[1]: box[1] + box[3], box[0]: box[0] + box[2], :
            ]
        width, height = img.shape[:2]
        scale = (width / newsize[0], height / newsize[0])
        #     print(scale)
        boxes = []
        for frame, box in zip(image_sample["frames"], image_sample["boxes"]):
            frame_path = root_path + frame
            box = [
                box[0] / scale[1],
                box[1] / scale[0],
                box[2] / scale[1],
                box[3] / scale[0],
            ]
            box = list(map(int, box))
            boxes.append(box)
            img = cv2.imread(frame_path)
            img = cv2.resize(img, newsize)

            os.makedirs(Path(save_path + frame).parent, exist_ok=True)

            cv2.imwrite(save_path + frame, img)
        cv2.imwrite((save_path + frame).replace(".jpg", "_cropped.jpg"), croped_image)
        #     print(boxes)
        track_ids_v2[track_id] = {
            "frames": image_sample["frames"],
            "boxes": boxes,
            "nl": image_sample["nl"],
        }
    with open(save_path + file_path.split("/")[-1], "w") as f_w:
        json.dump(track_ids_v2, f_w)


if __name__ == "__main__":
    prepare_data(
        file_path="./raw_data/AIC21_Track5_NL_Retrieval/data/train-tracks.json"
    )
    prepare_data(file_path="./raw_data/AIC21_Track5_NL_Retrieval/data/test-tracks.json")
    with open(
        "./raw_data/AIC21_Track5_NL_Retrieval/data/test-queries.json", "r"
    ) as f_r:
        test_queries = json.load(f_r)

    with open("./extracted_data/test-queries.json", "w") as f_w:
        json.dump(test_queries, f_w)
