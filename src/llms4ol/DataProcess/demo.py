import json
geotype_dataset_path = "../../assets/Datasets/SubTaskB.1-GeoNames/geoname_train_pairs.json"
with open(geotype_dataset_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
all_types = []

for item in data:
    # 将 parent 和 child 的值添加到列表中
    all_types.extend(item['parent'].split(', '))
    all_types.extend(item['child'].split(', '))

# 去重
unique_types = list(set(all_types))

# 打印结果
print(len(unique_types))