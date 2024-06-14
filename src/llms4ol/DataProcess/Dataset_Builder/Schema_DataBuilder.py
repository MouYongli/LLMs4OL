from llms4ol.path import find_root_path
from tqdm import tqdm
import json
import re
from collections import defaultdict
import random


#Process pretrain dataset (text only)
def Schema_Pretrain_dataset_builder(jaon_path):
    pass

def Schema_TaskA_CausalLM_dataset_builder(json_path):
    pass
    #

def combinations(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Construct mappings for parent-child and child-parent relationships
    pair1 = defaultdict(set)
    pair2 = defaultdict(set)
    pos_labeled_data = []
    neg_labeled_data = []

    for item in data:
        parent = item["parent"]
        child = item["child"]
        pair1[parent].add(child)
        pair1[child].add(parent)
        pos_labeled_data.append({"parent": item["parent"], "child": item["child"], "label": 1})
    

    # Generate all possible parent-child combinations
    all_entities = set(pair1.keys()).union(set(pair2.keys()))
    all_entities = list(all_entities)
    number = 2
    while len(set((parent, child) for parent in all_entities[:number] for child in all_entities[:number] if parent != child)) < len(data) * 10:
        number += 1
    all_possible_relations = set((parent, child) for parent in all_entities[:number] for child in all_entities[:number] if parent != child)

    parent_to_children = defaultdict(set)
    child_to_parents = defaultdict(set)
    for item in all_entities[:number]:
        for a in pair1[item]:
            parent_to_children[item].add(a)
        for b in pair2[item]:
            child_to_parents[item].add(b)

    # Generate new relationships based on transitive and hierarchical properties
    new_relations = set()

    # Generate new relationships according to the given rules
    for parent in parent_to_children:
        for child in parent_to_children[parent]:
            # Rule 1: If A is the parent of B, and B is the parent of C, then A is the parent of C
            if child in parent_to_children:
                for grandchild in parent_to_children[child]:
                    new_relations.add((parent, grandchild))
            # Rule 2: If A is the child of B, and B is the child of C, then A is the child of C
            if parent in child_to_parents:
                for grandparent in child_to_parents[parent]:
                    new_relations.add((grandparent, child))

    #Append newly generated relationships with label 1
    for parent, child in new_relations:
        pos_labeled_data.append({"parent": parent, "child": child, "label": 1})

    # Identify non-existent parent-child relationships and assign label 0
    existing_relations = set((item["parent"], item["child"]) for item in pos_labeled_data)
    non_existing_relations = all_possible_relations - existing_relations

    for parent, child in non_existing_relations:
        neg_labeled_data.append({"parent": parent, "child": child, "label": 0})


    # calculate spilt point (train: 0.8, eval: 0.2)
    split_point = int(0.8 * len(data))
    # pos : neg == 1 : 10
    train = pos_labeled_data + neg_labeled_data[:split_point*10]
    evaluation = pos_labeled_data[split_point:] + neg_labeled_data[split_point*10:]

    return train,evaluation

# the most complex method to build dataset: all combinations relation of types + context
def m1_Schema_TaskB_TextClf_train_dataset_builder(json_path):
    root_path = find_root_path()
    context_filename = root_path + f'/src/assets/Datasets/SubTaskB.2-Schema.org/processed/schemaTypes_processed.json'
    # open collected context file
    with open(context_filename, 'r', encoding='utf-8') as file:
        context_data = json.load(file)
    context_data_dict =  {item["term"].lower(): item["term_info"] for item in context_data }
    # open train-dataset file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("Schema.org training dataset size is: " + str(len(data)))
    train,evaluation = combinations(json_path)
    label = []
    text = []
    context = []
    for item in train:
        s = int(item["label"])
        parent = item["parent"]
        child = item["child"]
        parent_info = ""
        child_info = ""
        if parent.lower() in context_data_dict:
            parent_info = context_data_dict[parent.lower()]
        if child.lower() in context_data_dict:
            child_info = context_data_dict[child.lower()]
        label.append(s)
        text.append(f"{parent} is the superclass of {child}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is a subclass of {parent}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
        label.append(s)
        text.append(f"{parent} is the parent class of {child}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is a child class of {parent}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
        label.append(s)
        text.append(f"{parent} is a supertype of {child}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is a subtype of {parent}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
        label.append(s)
        text.append(f"{parent} is an ancestor class of {child}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is a descendant class of {parent}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
    return label, text, context

# two positiv / negative examples for each given train data (parent-child & child-parent relation) + context
def m2_Schema_TaskB_TextClf_train_dataset_builder(json_path):
    root_path = find_root_path()
    context_filename = root_path + f'/src/assets/Datasets/SubTaskB.2-Schema.org/processed/schemaTypes_processed.json'
    # open collected context file
    with open(context_filename, 'r', encoding='utf-8') as file:
        context_data = json.load(file)
    context_data_dict =  {item["term"].lower(): item["term_info"] for item in context_data }
    # open train-dataset file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("Schema.org training dataset size is: " + str(len(data)))
    train,evaluation = combinations(json_path)
    label = []
    text = []
    context = []
    for item in train:
        s = int(item["label"])
        parent = item["parent"]
        child = item["child"]
        parent_info = ""
        child_info = ""
        if parent.lower() in context_data_dict:
            parent_info = context_data_dict[parent.lower()]
        if child.lower() in context_data_dict:
            child_info = context_data_dict[child.lower()]
        label.append(s)
        text.append(f"{parent} is the superclass / parent / supertype / ancestor class of {child}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is the subclass / child / subtype / descendant class of {parent}")
        context.append(
            f"There are two terms used to describe web page content and online resources: ##'{parent}':{parent_info} ##'{child}':{child_info}")
    return label, text, context

# all combinations relation of types -- variante 1
def m3_Schema_TaskB_TextClf_train_dataset_builder(json_path):
    # open train-dataset file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("Schema.org training dataset size is: " + str(len(data)))
    train,evaluation = combinations(json_path)
    label = []
    text = []
    context = []
    for item in train:
        s = int(item["label"])
        parent = item["parent"]
        child = item["child"]

        label.append(s)
        text.append(f"{parent} is the superclass / parent / supertype / ancestor class of {child}")
        context.append("")
        label.append(s)
        text.append(f"{child} is the subclass / child / subtype / descendant class of {parent}")
        context.append("")
    return label, text, context

# all combinations relation of types -- variante 2
def m4_Schema_TaskB_TextClf_train_dataset_builder(json_path):
    # open train-dataset file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("Schema.org training dataset size is: " + str(len(data)))
    train,evaluation = combinations(json_path)
    label = []
    text = []
    context = []
    for item in train:
        s = int(item["label"])
        parent = item["parent"]
        child = item["child"]
        label.append(s)
        text.append(f"{parent} is the superclass of {child}")
        context.append("")
        label.append(s)
        text.append(f"{child} is a subclass of {parent}")
        context.append("")
        label.append(s)
        text.append(f"{parent} is the parent class of {child}")
        context.append("")
        label.append(s)
        text.append(f"{child} is a child class of {parent}")
        context.append("")
        label.append(s)
        text.append(f"{parent} is a supertype of {child}")
        context.append("")
        label.append(s)
        text.append(f"{child} is a subtype of {parent}")
        context.append("")
        label.append(s)
        text.append(f"{parent} is an ancestor class of {child}")
        context.append("")
        label.append(s)
        text.append(f"{child} is a descendant class of {parent}")
        context.append("")
    return label, text, context


def Schema_TaskB_TextClf_test_dataset_builder(json_path):
    filename = json_path
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    label = []
    text = []
    label_mapper = {"correct": 1, "incorrect": 0}
    #positive examples
    for item in data:
        parent = item["text_a"]
        child = item["text_b"]
        content_label = item["label"]
        label.append(label_mapper[content_label])
        text.append(f"{parent} is the superclass of {child}")
    return label, text
    

def Schema_TaskB_TextClf_evl_dataset_builder(json_path):
    train,evaluation = combinations(json_path)
    label = []
    text = []
    context = []
    # generate eval datasets
    for item in evaluation:
        s = int(item["label"])
        parent = item["parent"]
        child = item["child"]
        label.append(s)
        text.append(f"{parent} is the superclass of {child}")
        context.append("")
        label.append(s)
        text.append(f"{child} is a subclass of {parent}")
        context.append("")
        label.append(s)
        text.append(f"{parent} is the parent class of {child}")
        context.append("")
        label.append(s)
        text.append(f"{child} is a child class of {parent}")
        context.append("")
        label.append(s)
        text.append(f"{parent} is a supertype of {child}")
        context.append("")
        label.append(s)
        text.append(f"{child} is a subtype of {parent}")
        context.append("")
        label.append(s)
        text.append(f"{parent} is an ancestor class of {child}")
        context.append("")
        label.append(s)
        text.append(f"{child} is a descendant class of {parent}")
        context.append("")
    
    return label, text, context


