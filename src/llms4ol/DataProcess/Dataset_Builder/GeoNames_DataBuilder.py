from llms4ol.path import find_root_path
from tqdm import tqdm
import json
import re,csv
from collections import defaultdict
import random

def merge_info_files():
    collected_info = []
    for num in range(100):
        file_path = f"/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskA.2-GeoNames/processed/geo_data_part{num}.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for item in data:
            collected_info.append(item)
    output_path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskA.2-GeoNames/term_info_parts.json"
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(len(collected_info))

def hierarchy_type():

    csv_path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.1-GeoNames/feature_codes.csv"
    L1_list = set()
    L1_to_L2_dict = {}
    L2_to_L1_dict = {}
    # 打开 CSV 文件
    with open(csv_path, mode='r', encoding='utf-8') as file:
        # 创建一个 CSV 读取器
        csv_reader = csv.reader(file)
        next(csv_reader)
        # 遍历每一行
        for row in csv_reader:
            # 这里可以添加处理每一行数据的代码
            # 例如，打印每一行的第一个元素
            L1=row[3].lower()
            L2=row[5].lower()
            list_data = eval(L1)
            # 使用 join 方法将列表转换为逗号分隔的字符串
            output_string = ','.join(list_data)
            L1_list.add(output_string)
            if output_string not in L1_to_L2_dict:
                L1_to_L2_dict[output_string] = set()
            # 将 L2 添加到 L2_dict 中对应的 set
            L1_to_L2_dict[output_string].add(L2)
            if L2 not in L2_to_L1_dict:
                L2_to_L1_dict[L2] = ""
            L2_to_L1_dict[L2] = output_string
    print(f"The number of Level 1 Types is {len(L1_to_L2_dict)}, the number of Level 2 Types is {len(L2_to_L1_dict)}")
    return list(L1_list),L1_to_L2_dict,L2_to_L1_dict

#Process pretrain dataset (text only)
def Geo_Pretrain_dataset_builder():
    training_file = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskA.2-GeoNames/geonames_train.json"
    term_info = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskA.2-GeoNames/term_info_parts.json"
    type_info = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.1-GeoNames/geoTypes_processed.json"
    with open(training_file, 'r', encoding='utf-8') as file:
        training_data = json.load(file)
    with open(term_info, 'r', encoding='utf-8') as file:
        term_data = json.load(file)
    with open(type_info, 'r', encoding='utf-8') as file:
        type_data = json.load(file)
    term2info_dict = {item["term"]:item["term_info"] for item in term_data}
    type2info_dict = {item["term"]:item["term_info"] for item in type_data}
    L1_list,L1_to_L2_dict,L2_to_L1_dict = hierarchy_type()
    pretrain_prompt = []
    for item in training_data[:int(len(training_data)/100)]:
        term = item["term"]
        term_info = ""
        if term in term2info_dict:
            term_info = term2info_dict[term]
        L1 = L2_to_L1_dict[item["type"].lower()]
        L1_info = ""
        for s_type in L1.split(","):
            if s_type in type2info_dict:
                L1_info += type2info_dict[s_type]
        L2 = item["type"].lower()
        L2_info = ""
        for s_type in L2.split(","):
            if s_type in type2info_dict:
                L2_info += type2info_dict[s_type]
        text = f'''
        The term "{term}" from the GeoNames dataset. {term_info}. It falls under the top-level classification of "{L1}".
        Given this description, it can be logically inferred that "{term}" should belong to the specific sub-category "{L2}" within this top-level classification.
        "{L2}" is described as: {L2_info}
        Consequently, based on this inference, the type of this term is determined to be "{L2}".
        '''
        pretrain_prompt.append(text.replace("\n",""))
    print(len(pretrain_prompt))
    return pretrain_prompt



def Geo_TaskA_CausalLM_dataset_builder(json_path):
    root_path = find_root_path()
    typing_file = root_path + "/src/assets/Datasets/SubTaskA.2-GeoNames/geonames_train.json"

def Geo_TaskA_TextClf_dataset_builder():
    root_path = find_root_path()
    json_file = root_path + "/src/assets/Datasets/SubTaskA.2-GeoNames/geonames_train.json"
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
    label = []
    for item in data:
        text.append(str(item["term"]))
        label.append(label2id[item["type"]])
    print("GeoNames")
    print("The total number of data for training is: ",len(text))
    print("The total number of labels is: ",len(id2label))
    return id2label,label2id,text,label


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
def m1_Geo_TaskB_TextClf_train_dataset_builder(json_path):
    root_path = find_root_path()
    context_filename = root_path + f'/src/assets/Datasets/SubTaskB.1-GeoNames/geoTypes_processed.json'
    #open collected context file
    with open(context_filename, 'r', encoding='utf-8') as file:
        context_data = json.load(file)
    # open train-dataset file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("GeoNames training dataset size is: " + str(len(data)))
    train,evaluation  = combinations(json_path)
    label = []
    text = []
    context = []
    for item in train:
        s = int(item["label"])
        parent = item["parent"]
        child = item["child"]
        p_list = parent.split(",")
        p_info_list=[]
        for a in p_list:
            a = re.sub(r'\s+', '', a)
            for i in context_data:
                if a.lower() in i["term"].lower():
                    p_info_list.append(i["term_info"])
                    break
        parent_info = ".".join(p_info_list)

        c_list = child.split(",")
        c_info_list=[]
        for b in c_list:
            b = re.sub(r'\s+', '', b)
            for i in context_data:
                if b.lower() in i["term"].lower():
                    c_info_list.append(i["term_info"])
                    break
        child_info = ".".join(c_info_list)

        label.append(s)
        text.append(f"{parent} is the superclass of {child}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is a subclass of {parent}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
        label.append(s)
        text.append(f"{parent} is the parent class of {child}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is a child class of {parent}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
        label.append(s)
        text.append(f"{parent} is a supertype of {child}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is a subtype of {parent}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
        label.append(s)
        text.append(f"{parent} is an ancestor class of {child}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is a descendant class of {parent}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
    return label, text, context

# two positiv / negative examples for each given train data (parent-child & child-parent relation) + context
def m2_Geo_TaskB_TextClf_train_dataset_builder(json_path):
    root_path = find_root_path()
    context_filename = root_path + f'/src/assets/Datasets/SubTaskB.1-GeoNames/geoTypes_processed.json'
    #open collected context file
    with open(context_filename, 'r', encoding='utf-8') as file:
        context_data = json.load(file)
    # open train-dataset file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("GeoNames training dataset size is: " + str(len(data)))
    train,evaluation = combinations(json_path)
    label = []
    text = []
    context = []
    for item in train:
        s = int(item["label"])
        parent = item["parent"]
        child = item["child"]
        p_list = parent.split(",")
        p_info_list=[]
        for a in p_list:
            a = re.sub(r'\s+', '', a)
            for i in context_data:
                if a.lower() in i["term"].lower():
                    p_info_list.append(i["term_info"])
                    break
        parent_info = ".".join(p_info_list)

        c_list = child.split(",")
        c_info_list=[]
        for b in c_list:
            b = re.sub(r'\s+', '', b)
            for i in context_data:
                if b.lower() in i["term"].lower():
                    c_info_list.append(i["term_info"])
                    break
        child_info = ".".join(c_info_list)

        label.append(s)
        text.append(f"{parent} is the superclass / parent / supertype / ancestor class of {child}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
        label.append(s)
        text.append(f"{child} is the subclass / child / subtype / descendant class of {parent}")
        context.append(
            f"They are two geographical terms.'{parent}':{parent_info} '{child}':{child_info}")
    return label, text, context

# all combinations relation of types -- variante 1
def m3_Geo_TaskB_TextClf_train_dataset_builder(json_path):
    # open train-dataset file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("GeoNames training dataset size is: " + str(len(data)))
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
def m4_Geo_TaskB_TextClf_train_dataset_builder(json_path):
    # open train-dataset file
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("GeoNames training dataset size is: " + str(len(data)))
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

def Geo_TaskB_TextClf_test_dataset_builder(json_path):
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
    

def Geo_TaskB_TextClf_evl_dataset_builder(json_path):
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

