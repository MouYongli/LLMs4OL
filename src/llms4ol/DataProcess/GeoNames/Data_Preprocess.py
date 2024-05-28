from llms4ol.DataProcess.GeoNames.Context_Provider import *
from llms4ol.path import find_root_path
from tqdm import tqdm
import multiprocessing,json
import time


# Merge all data from multi_subprocess into one single json file
def pretrain_json_data_merge(task_type,num):
    print("Now start merging all json files into one single file")
    root_path = find_root_path()
    if task_type == "A":
        merged_path = root_path + f'/src/assets/Datasets/SubTaskA.2-GeoNames/processed/geoNames_processed.json'
        json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskA.2-GeoNames/processed/geo_data_part{i}.json' for i in range(num)]
    else:
        merged_path = root_path + f'/src/assets/Datasets/SubTaskB.1-GeoNames/processed/geoTypes_processed.json'
        json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.1-GeoNames/processed/geo_type_part{i}.json' for i in range(num)]
    data_list = []
    for file_path in json_files_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                data_list.append(item)
    #Remove all unnecessary info
    parts_to_remove = ["couldn't find any","does not require","assist you further","feel free to","already in English","require further","any additional information"]
    for item in data_list:
        info_list = item["term_info"].split(".")
        for part in parts_to_remove:
            for i in range(len(info_list)):
                if part in info_list[i]:
                    info_list[i] = ""
                    break
        result = ".".join(info_list)
        result = result.replace("in English", "")
        item["term_info"] = result
    with open(merged_path, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=4)
    print(len(data_list))
    print("Merge Finished!")

def tqdm_subprocessing_module(num,data,task_type):
    current = data[0]["id"]
    max = len(data)-1
    data_id = 0
    root_path = find_root_path()
    if task_type == "A":
        output_path = root_path + f'/src/assets/Datasets/SubTaskA.2-GeoNames/processed/geo_data_part{num}.json'
    else:
        output_path = root_path + f'/src/assets/Datasets/SubTaskB.1-GeoNames/processed/geo_type_part{num}.json'
    #Check files in local before inference. Records Recovery
    try:
        with open(output_path, 'r', encoding='utf-8') as file:
            exist_data = json.load(file)
        if len(exist_data) > 0 :
            start_id = exist_data[-1]["id"]+1
            current = start_id - current
            data[:len(exist_data)] = exist_data
        else:
            current = 0
    except (FileNotFoundError, json.JSONDecodeError):
        current = 0
    p_bar = tqdm(total = max + 1, desc=f"Subprocess {num} Progress", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    p_bar.update(current)
    p_bar_temp = 0
    while current <= max:
        try:
            data_id = 0
            for i, item in enumerate(data):
                #Skip all data already processed
                if i < current:
                    data_id +=1
                    continue
                else:
                    item['term_info'] = GPT_Inference_For_GeoLocations(item["term"])
                    data_id += 1
                    current += 1
                    p_bar_temp += 1
                    if p_bar_temp % 100 == 0:
                         p_bar.update(p_bar_temp)
                         p_bar_temp = 0
                #Save results every 10 records
                if i % 10 == 1:
                    with open(output_path, 'w', encoding='utf-8') as file:
                        json.dump(data[:data_id], file, ensure_ascii=False, indent=4)
        except Exception as e:
            #print(f"Process {num} has Error")
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(data[:data_id], file, ensure_ascii=False, indent=4)
            time.sleep(5)
    p_bar.close()
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"JSON file {num} has saved! ")


#####################################################################
#TASK A

#For Pretrain
def preprocess_geoTerms_for_all():
    root_path = find_root_path()
    json_file = root_path + "/src/assets/Datasets/SubTaskA.2-GeoNames/geonames_train.json"
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    id = 0
    print("Data loading initialization")
    for item in tqdm(data, desc="GeoTerms Loading", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        new_item = {'id': id, 'term': item['term'], 'term_info': ""}
        item.clear()
        item.update(new_item)
        id += 1
    parent, child = extract_geoType_to_array()

    # Split comma-separated types in parent_type
    all_types = parent + child
    unique_type = []
    for a in all_types:
        types = a.split(",")
        for b in types:
            unique_type.append(b)
    unique_type = list(set(unique_type))
    print(f'There are in total {len(unique_type)} unique type after spliting and de-duplicating')
    for item in tqdm(unique_type, desc="GeoTypes Loading", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        term_info = ""
        data.append({
        "id": id,
        "term": item,
        "term_info": term_info
    })
        id += 1
    # data size: 8079537 
    print("Data loading finished")

    total_items = len(data)
    fields_per_part = total_items // 100
    parts = [[] for _ in range(100)]
    for i, item in enumerate(data):
        part_index = i // fields_per_part
        if i >= fields_per_part * 100:
            part_index = 99 # ensure the last part contains all remain
        parts[part_index].append(item)

    #Collect relevant contexts with multiple processes
    processes = []
    #Multiprocess
    for i, part in enumerate(parts):
        process = multiprocessing.Process(target=tqdm_subprocessing_module, args=(i,part,"A",))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    print("All workers have finished")
    pretrain_json_data_merge("A",100)


#####################################################################
#TASK B

def extract_geoType_to_array():
    root_path = find_root_path()
    json_file = root_path + "/src/assets/Datasets/SubTaskB.1-GeoNames/geoname_train_pairs.json"
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    super = set()
    sub = set()

    for item in data:
        super.add(item["parent"])
        sub.add(item["child"])

    return list(super),list(sub)

#For Finetuning
def preprocess_geoTypes():
    parent, child = extract_geoType_to_array()
    # Split comma-separated types in parent_type
    all_types = parent + child
    unique_type = []
    for a in all_types:
        types = a.split(",")
        for b in types:
            unique_type.append(b)
    unique_type = list(set(unique_type))
    print(f'There are in total {len(unique_type)} unique type after spliting and de-duplicating')
    data = []
    id = 0
    for item in tqdm(unique_type, desc="GeoTypes Loading", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        term_info = ""
        data.append({
        "id": id,
        "term": item,
        "term_info": term_info
    })
        id += 1
    print("Data loading finished")

    total_items = len(data)

    parts = [[] for _ in range(100)]
    part_id = 0
    while data:
        parts[part_id].append(data.pop())
        part_id += 1
        if part_id == 100:
            part_id = 0
    
    #Collect relevant contexts with multiple processes
    processes = []
    #Multiprocess
    for i, part in enumerate(parts):
        process = multiprocessing.Process(target=tqdm_subprocessing_module, args=(i,part,"B",))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    print("All workers have finished")
    #Merge 100 json files into one file
    pretrain_json_data_merge("B",100)