from collections import defaultdict,deque
from llms4ol.DataProcess.Dataset_Builder import GeoNames_DataBuilder,GO_DataBuilder,Schema_DataBuilder,UMLS_DataBuilder,WordNet_DataBuilder

def test(num):
    print(num)
    # Original dataset
    data = [
        {"parent": "A", "child": "B"},
        {"parent": "C", "child": "D"},
        {"parent": "C", "child": "E"},
        {"parent": "D", "child": "F"},
    ]

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
    while len(set((parent, child) for parent in all_entities[:number] for child in all_entities[:number] if parent != child)) < len(data):
        number += 1
    all_possible_relations = set((parent, child) for parent in all_entities[:number] for child in all_entities[:number] if parent != child)

    parent_to_children = defaultdict(set)
    child_to_parents = defaultdict(set)
    for item in all_entities[:number]:
        for i in pair1[item]:
            parent_to_children[item].add(i)
        child_to_parents[item].add(tuple(pair2[item]))

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

r = ~(1<<4)
print(f"{r: 08b}")

# 定义带符号的二进制数
binary_str = '-0010001'

# 检查负号
if binary_str[0] == '-':
    is_negative = True
    binary_str = binary_str[1:]  # 去掉负号
else:
    is_negative = False

# 将二进制字符串转换为整数
decimal_value = int(binary_str, 2)

# 如果是负数，显示正数形式
if is_negative:
    # 获取这个二进制数的位数
    bit_length = len(binary_str)
    print(bit_length)
    # 计算补码
    max_value = (1 << bit_length)  # 2^bit_length
    positive_decimal_value = max_value - decimal_value
    
    print(f"原始二进制数: -{binary_str}")
    print(f"正数形式: {bin(positive_decimal_value)}")
else:
    print(f"原始二进制数: {binary_str}")
    print(f"正数形式: {bin(decimal_value)}")

