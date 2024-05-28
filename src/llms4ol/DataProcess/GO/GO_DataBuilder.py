from llms4ol.DataProcess.GO.Context_Provider import *
from llms4ol.path import find_root_path
from tqdm import tqdm
import re,json


#Process pretrain dataset (text only)
def Pretrain_dataset_builder():
    pass

def FinetuneA_dataset_builder():
    pass
    #

def FintuneB_dataset_builder(json_path):
    root_path = find_root_path()
    context_filename = root_path + f'/src/assets/Datasets/SubTaskB.4-GO/processed/geoTypes_processed.json'
    #open collected context file
    with open(context_filename, 'r', encoding='utf-8') as file:
        context_data = json.load(file)
    # open train-dataset file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    label = []
    text = []
    context = []
    for item in data:
        # positive & negative examples
        for s in [0,1]:
            if s == 0:
                child = item["parent"]
                parent = item["child"]
            else:
                parent = item["parent"]
                child = item["child"]
            p_list = parent.split(",")
            p_info_list=[]
            for a in p_list:
                a = re.sub(r'\s+', '', a)
                for i in context_data:
                    if a in i["term"]:
                        p_info_list.append(i["term_info"])
                        break
            parent_info = ".".join(p_info_list)

            c_list = child.split(",")
            c_info_list=[]
            for b in c_list:
                b = re.sub(r'\s+', '', b)
                for i in context_data:
                    if b in i["term"]:
                        c_info_list.append(i["term_info"])
                        break
            child_info = ".".join(c_info_list)

            label.append(s)
            text.append(f"{parent} is the superclass of {child}")
            context.append(
                f"There are two geographical names: {parent} and {child}.{parent_info}{child_info}")
            label.append(s)
            text.append(f"{child} is a subclass of {parent}")
            context.append(
                f"There are two geographical names: {parent} and {child}. {parent_info}{child_info}")
            label.append(s)
            text.append(f"{parent} is the parent class of {child}")
            context.append(
                f"There are two geographical names: {parent} and {child}. {parent_info}{child_info}")
            label.append(s)
            text.append(f"{child} is a child class of {parent}")
            context.append(
                f"There are two geographical names: {parent} and {child}. {parent_info}{child_info}")
            label.append(s)
            text.append(f"{parent} is a supertype of {child}")
            context.append(
                f"There are two geographical names: {parent} and {child}. {parent_info}{child_info}")
            label.append(s)
            text.append(f"{child} is a subtype of {parent}")
            context.append(
                f"There are two geographical names: {parent} and {child}. {parent_info}{child_info}")
            label.append(s)
            text.append(f"{parent} is an ancestor class of {child}")
            context.append(
                f"There are two geographical names: {parent} and {child}. {parent_info}{child_info}")
            label.append(s)
            text.append(f"{child} is a descendant class of {parent}")
            context.append(
                f"There are two geographical names: {parent} and {child}. {parent_info}{child_info}")
    return label, text, context


def FintuneB_evl_dataset_builder(json_path):
    filename = json_path
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    label = []
    text = []
    context = []
    label_mapper = {"correct": 1, "incorrect": 0}
    #positive examples
    for item in data:
        parent = item["text_a"]
        child = item["text_b"]
        content_label = item["label"]
        label.append(label_mapper[content_label])
        text.append(f"{parent} is the superclass of {child}")
    return label, text

#preprocess_geoTypes()


