from llms4ol.DataProcess.Schema.Context_Provider import *
from llms4ol.path import find_root_path
from tqdm import tqdm
import multiprocessing,json
import time

def extract_GOType_to_array():
    root_path = find_root_path()
    json_file = root_path + "/src/assets/Datasets/SubTaskB.4-GO/go_train_pairs.json"
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    entities = set()

    for item in data:
        entities.add(item["parent"])
        entities.add(item["child"])

    entities_list = list(entities)

    return entities_list

#####################################################################
#TASK A


#####################################################################
#TASK B

#For Finetuning
def preprocess_GO_Types():
    terms = extract_GOType_to_array()
    print(f'There are in total {len(terms)} unique type')
    data = []
    id = 0
    for item in tqdm(terms, desc="GOTypes Loading", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
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
        process = multiprocessing.Process(target=tqdm_subprocessing_module, args=(i,part,))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    print("All workers have finished")
    #Merge 100 json files into one file
    json_data_merge(100)


# Merge all data from multi_subprocess into one single json file
def json_data_merge(num):
    print("Now start merging all json files into one single file")
    root_path = find_root_path()
    merged_path = root_path + f'/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.2-Schema.org/processed/geoTypes_processed.json'
    json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.2-Schema.org/processed/geo_type_part{i}.json' for i in range(num)]
    data_list = []
    for file_path in json_files_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                data_list.append(item)
    #remove all unnecessary info
    parts_to_remove = ["couldn't find any","does not require","assist you further","feel free to","require further","any additional information","don't have information","I'm sorry,"]
    for item in data_list:
        info_list = item["term_info"].split(".")
        for part in parts_to_remove:
            for i in range(len(info_list)):
                if part in info_list[i]:
                    info_list[i] = ""
                    break
        result = ".".join(info_list)
        item["term_info"] = result
    with open(merged_path, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=4)
    print(len(data_list))
    print("Merge Finished!")

def tqdm_subprocessing_module(num,data):
    current = data[0]["id"]
    max = len(data)-1
    data_id = 0
    root_path = find_root_path()
    output_path = root_path + f'/src/assets/Datasets/SubTaskB.2-Schema.org/processed/Schema_type_part{num}.json'
    #check files in local before inference. Records Recovery
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
                    item['term_info'] = GPT_Inference_For_Schema_TaskB(item["term"])
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

