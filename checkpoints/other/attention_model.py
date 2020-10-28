import json
import os
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_path_0 = 'C:\\Users\\20522516\\phd\\RESEARCH\\CYBER_DATASETS\\DATASET_DEEP_WEB\\labels_0.json'
file_path_1 = 'C:\\Users\\20522516\\phd\\RESEARCH\\CYBER_DATASETS\\DATASET_DEEP_WEB\\labels_1.json'

list_0_image_desc = []
list_1_image_desc = []

with open(file_path_0, encoding="utf8") as json_file:
    data = json.load(json_file)
    for entry in data:
        details = {"name": entry, "meta": []}
        for meta_data in data[entry]:
            if len(meta_data.split("=")) > 1:
                type_image = meta_data.split("=")[0]
                type_detail = meta_data.split("=")[1]
                details["meta"].append((type_image, type_detail))
            else:
                details["meta"].append(("other", meta_data))

        list_0_image_desc.append(details)

with open(file_path_1, encoding="utf8") as json_file:
    data = json.load(json_file)
    for entry in data:
        details = {"name": entry, "meta": []}
        for meta_data in data[entry]:
            if len(meta_data.split("=")) > 1:
                type_image = meta_data.split("=")[0]
                type_detail = meta_data.split("=")[1]
                details["meta"].append((type_image, type_detail))
            else:
                details["meta"].append(("other", meta_data))

        list_1_image_desc.append(details)

print(list_0_image_desc)
print(list_1_image_desc)
print(len(list_0_image_desc))
print(len(list_1_image_desc))
#
# with open(file_path_1, encoding="utf8") as json_file:
#     data = json.load(json_file)