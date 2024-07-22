import json
def count():
    file_path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.2-Schema.org/processed/schemaTypes_processed.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        exist_data = json.load(file)
    print(len(exist_data))
count()