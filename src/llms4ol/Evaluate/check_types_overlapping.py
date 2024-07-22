import json

########过拟合数据合并###
def taskA_overlapping_types():
    train_dataset = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskA.3-UMLS/nci_train.json"
    ######
    #    {
    #    "ID": "C0265964-10065",
    #    "term": "mutilating keratoderma of Vohwinkel (diagnosis)",
    #    "type": [
    #        "congenital abnormality"
    #    ]
    #}
    ######
    test_dataset = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/A.3(FS)_UMLS_NCI_Test.json"
    ######
    #    {
    #    "ID": "C0000817-12",
    #    "term": "sepsis following abortion (diagnosis)"
    #},
    ######
    # 读取第一个JSON文件
    with open(train_dataset, 'r') as f1:
        data1 = json.load(f1)

    # 读取第二个JSON文件
    with open(test_dataset, 'r') as f2:
        data2 = json.load(f2)

    # 创建一个字典，将ID映射到第一个文件中的"type"
    type_dict = {entry['ID']: entry['type'] for entry in data1}
    # 遍历第二个JSON文件并更新"type"字段
    count = 0
    for entry in data2:
        if entry['ID'] in type_dict:
            count += 1
            entry['type'] = type_dict[entry['ID']]
    print(count)
    
#overfit_merge()

def taskB_overlapping_types():
    train_dataset = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.1-GeoNames/geoname_train_types.txt"
    test_dataset = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/B.1(FS)_GeoNames_Test.txt"
    # 读取第一个JSON文件
    with open(train_dataset, 'r') as f1:
        data1 = json.load(f1)

    # 读取第二个JSON文件
    with open(test_dataset, 'r') as f2:
        data2 = json.load(f2)