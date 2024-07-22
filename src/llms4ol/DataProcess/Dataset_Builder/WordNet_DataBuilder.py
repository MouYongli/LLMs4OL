from llms4ol.path import find_root_path
from tqdm import tqdm
import json
import re
from collections import defaultdict
import random


def WN_TaskA_TextClf_dataset_builder():
    root_path = find_root_path()
    json_file = root_path + "/src/assets/Datasets/SubTaskA.1-WordNet/wordnet_train.json"
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    types = set()
    #de-duplicate
    for item in data:
        types.add(item["type"])
    labels = list(types)
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    text = []
    sentences = []
    label = []
    for item in data:
        sentences.append(str(item["sentence"]))
        text.append(str(item["term"]))
        label.append(label2id[item["type"]])
    print("WordNet")
    print("The total number of data for training is: ",len(text))
    print("The total number of labels is: ",len(id2label))
    return id2label,label2id,text,label,sentences