from llms4ol.DataProcess.Context_Provider import *
from llms4ol.path import find_root_path
from tqdm import tqdm
from pathlib import Path
import multiprocessing,json
import time
import re,os
from collections import Counter

root_path = find_root_path()

#####################################################################
#TASK A


def term_type_extraction(json_file):
    return
def taskA_preprocess():
    pass



#####################################################################
#TASK B


parts_to_remove = ["couldn't find any","does not require","assist you further","feel free to","I'm currently unable","the search results","I'm unable to","recommend referring directly","bear with me","searching for the most relevant information","I'm currently checking the most relevant","already in English","require further","any additional information","already an English","don't have information","I'm sorry,","For further exploration","For more detailed information"]
def taskB_preprocess(args):
    #num is the number of multiprocesses
    num = 50
    root_path = find_root_path()
    if args.num == 1:
        dataset_name = "GeoNames"
        data_provider = GPT_Inference

        #train dataset path
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.1-GeoNames/geoname_train_pairs.json"

        # The only file we need in the end after preprocessing data
        main_processed_path = root_path + f'/src/assets/Datasets/SubTaskB.1-GeoNames/processed/geoTypes_processed.json'

        # All json files that hold intermediate results for first process (will be deleted after merging all parts files)
        json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.1-GeoNames/processed/geo_type_part{i}.json' for i in range(num)]


        execute_func(num, train_dataset_path, main_processed_path, json_files_paths, data_provider, dataset_name, args.task, args.num)

        #####################################################################
        #re-inference

        # File holds all merged info after re-inference
        re_inference_path = root_path + f'/src/assets/Datasets/SubTaskB.1-GeoNames/processed/GeoNames_Types_re_inference.json'

        # All json files that hold intermediate results for re_inference (will be deleted after merging all parts files)
        parts_re_inference_json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.1-GeoNames/processed/GeoNames_re_inference{i}.json' for i in range(num)]

        #loop in case there is still some info need to be re-inference
        count = 0
        while re_inference(num, main_processed_path, parts_re_inference_json_files_paths, re_inference_path, data_provider, dataset_name, args.task, args.num):
            count += 1
            if count > 5:
                break
        print("re_inference finished!")

    elif args.num == 2:
        dataset_name = "Schema.org"
        data_provider = GPT_Inference

        #train dataset path
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.2-Schema.org/schemaorg_train_pairs.json"

        # The only file we need in the end after preprocessing data
        main_processed_path = root_path + f'/src/assets/Datasets/SubTaskB.2-Schema.org/processed/schemaTypes_processed.json'

        # All json files that hold intermediate results for first process (will be deleted after merging all parts files)
        json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.2-Schema.org/processed/Schema_type_part{i}.json' for i in range(num)]


        execute_func(num, train_dataset_path, main_processed_path, json_files_paths, data_provider, dataset_name, args.task, args.num)

        #####################################################################
        #re-inference

        # File holds all merged info after re-inference
        re_inference_path = root_path + f'/src/assets/Datasets/SubTaskB.2-Schema.org/processed/Schema_Types_re_inference.json'

        # All json files that hold intermediate results for re_inference (will be deleted after merging all parts files)
        parts_re_inference_json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.2-Schema.org/processed/Schema_re_inference{i}.json' for i in range(num)]

        #loop in case there is still some info need to be re-inference
        count = 0
        while re_inference(num, main_processed_path, parts_re_inference_json_files_paths, re_inference_path, data_provider, dataset_name, args.task, args.num):
            count += 1
            if count > 5:
                break
        print("re_inference finished!")

    elif args.num == 3:
        dataset_name = "UMLS"
        data_provider = GPT_Inference

        #train dataset path
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.3-UMLS/umls_train_pairs.json"

        # The only file we need in the end after preprocessing data
        main_processed_path = root_path + f'/src/assets/Datasets/SubTaskB.3-UMLS/processed/umls_Types_processed.json'

        # All json files that hold intermediate results for first process (will be deleted after merging all parts files)
        json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.3-UMLS/processed/umls_type_part{i}.json' for i in range(num)]


        #execute_func(num, train_dataset_path, main_processed_path, json_files_paths, data_provider, dataset_name, args.task, args.num)

        #####################################################################
        #re-inference

        # File holds all merged info after re-inference
        re_inference_path = root_path + f'/src/assets/Datasets/SubTaskB.3-UMLS/processed/umls_Types_re_inference.json'

        # All json files that hold intermediate results for re_inference (will be deleted after merging all parts files)
        parts_re_inference_json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.3-UMLS/processed/UMLS_re_inference{i}.json' for i in range(num)]

        #loop in case there is still some info need to be re-inference
        count = 0
        while re_inference(num, main_processed_path, parts_re_inference_json_files_paths, re_inference_path, data_provider, dataset_name, args.task, args.num):
            count += 1
            if count > 5:
                break
        print("re_inference finished!")

    elif args.num == 4:
        dataset_name = "Gene Ontology"
        data_provider = GPT_Inference

        #train dataset path
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.4-GO/go_train_pairs.json"

        # The only file we need in the end after preprocessing data
        main_processed_path = root_path + f'/src/assets/Datasets/SubTaskB.4-GO/processed/GO_Types_processed.json'

        # All json files that hold intermediate results for first process (will be deleted after merging all parts files)
        json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.4-GO/processed/GO_type_part{i}.json' for i in range(num)]


        execute_func(num, train_dataset_path, main_processed_path, json_files_paths, data_provider, dataset_name, args.task, args.num)

        #####################################################################
        #re-inference

        # File holds all merged info after re-inference
        re_inference_path = root_path + f'/src/assets/Datasets/SubTaskB.4-GO/processed/GO_Types_re_inference.json'

        # All json files that hold intermediate results for re_inference (will be deleted after merging all parts files)
        parts_re_inference_json_files_paths = [root_path + f'/src/assets/Datasets/SubTaskB.4-GO/processed/GO_re_inference{i}.json' for i in range(num)]

        #loop in case there is still some info need to be re-inference
        count = 0
        while re_inference(num, main_processed_path, parts_re_inference_json_files_paths, re_inference_path, data_provider, dataset_name, args.task, args.num):
            count += 1
            if count > 5:
                break
        print("re_inference finished!")

def extract_types_to_array(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    entities = set()

    for item in data:
        entities.add(item["parent"])
        entities.add(item["child"])

    entities_list = list(entities)

    return entities_list


# Merge all data from multi_subprocess into one single json file
def json_data_merge(dataset_name:str, merged_path:Path, json_files_paths:list[str]):
    print("Now start merging all json files into one single file")
    data_list = []
    for file_path in json_files_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    
    #remove all unnecessary info
    parts_to_remove.append(dataset_name)
    for item in data:
        info_list = item["term_info"].split(".")
        for part in parts_to_remove:
            for i in range(len(info_list)):
                if part in info_list[i]:
                    info_list[i] = ""
                    break
        result = ".".join(info_list)
        cleaned_text = re.sub(r'\[\[\d+\]\]\(https?://[^\)]+\)', '', result)
        item["term_info"] = cleaned_text
    with open(merged_path, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=4)
    print(len(data_list))
    for path in json_files_paths:
        try:
            os.remove(path)
        except Exception as e:
            print(f"")
    print("Merge Finished!")

def subprocessing_module(num,data,output_path, methode, task, task_num, progress_dict):
    current = 0
    max = len(data)-1
    #check files in local before inference. Records Recovery
    try:
        with open(output_path, 'r', encoding='utf-8') as file:
            exist_data = json.load(file)
        if len(exist_data) > 0 :
            current = len(exist_data)-1
            data[:len(exist_data)] = exist_data
        else:
            current = 0
    except (FileNotFoundError, json.JSONDecodeError):
        current = 0
    progress_dict[num] = current
    while current <= max:
        try:
            data_idx = 0
            for i, item in enumerate(data):
                #Skip all data already processed
                if i < current:
                    data_idx +=1
                    continue
                else:
                    item['term_info'] = methode(task,task_num,item["term"])
                    data_idx += 1
                    current += 1
                    progress_dict[num] = current
                #Save results every 10 records
                if i % 10 == 1:
                    with open(output_path, 'w', encoding='utf-8') as file:
                        json.dump(data[:data_idx], file, ensure_ascii=False, indent=4)
        except Exception as e:
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(data[:data_idx], file, ensure_ascii=False, indent=4)
            time.sleep(5)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    progress_dict[num] = len(data)


#For Finetuning
def execute_func(num:int, train_dataset_path:Path, merged_path:Path, json_files_paths:list[str], data_provider, dataset_name:str, task:str, task_num:int):
    terms = extract_types_to_array(train_dataset_path)
    print(f'There are in total {len(terms)} unique type in {dataset_name} dataset')
    data = []
    id = 0
    tip = dataset_name + "Types Loading.."
    for item in tqdm(terms, desc = tip, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        term_info = ""
        data.append({
        "id": id,
        "term": item,
        "term_info": term_info
    })
        id += 1
    print("Data loading finished")

    parts = [[] for _ in range(num)]
    part_id = 0
    while data:
        parts[part_id].append(data.pop())
        part_id += 1
        if part_id == num:
            part_id = 0

    manager = multiprocessing.Manager()
    progress_dict = manager.dict()

    # init progress dict
    for i in range(len(parts)):
        progress_dict[i] = 0

    #Collect relevant contexts with multiple processes
    processes = []

    #Multiprocess
    for i, part in enumerate(parts):
        path = json_files_paths[i]
        process = multiprocessing.Process(target=subprocessing_module, args=(i,part,path,data_provider,task,task_num,progress_dict,))
        processes.append(process)
        process.start()

    with tqdm(total=len(terms)) as pbar:
        previous_progress = {i: 0 for i in range(len(parts))}
        while any(process.is_alive() for process in processes):
            current_progress = sum(progress_dict.values())
            pbar.update(current_progress - sum(previous_progress.values()))
            previous_progress = progress_dict.copy()
            time.sleep(1)

        while pbar.n < len(terms):
            current_progress = sum(progress_dict.values())
            pbar.update(len(terms) - pbar.n)
            time.sleep(1)

    for process in processes:
        process.join()

    pbar.close()
    print("All workers have finished")
    #Merge json files into one file
    json_data_merge(dataset_name, merged_path, json_files_paths)

#For the term that GPT can't genereta useful/clear context
def re_inference(num:int, merged_path:Path, json_files_paths:list[str], re_inference_path:Path, data_provider, dataset_name:str,  task:str, task_num:int):
    with open(merged_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(len(data))
    candidates = []
    for i, item in enumerate(data):
        #remove all unnecessary info
        parts_to_remove.append(dataset_name)
        info_list = item["term_info"].split(".")
        for part in parts_to_remove:
            for a in range(len(info_list)):
                if part in info_list[a]:
                    info_list[a] = ""
                    break
        result = ".".join(info_list)
        cleaned_text = re.sub(r'\[\[\d+\]\]\(https?://[^\)]+\)', '', result)
        item["term_info"] = cleaned_text
        if len(item["term_info"]) < 50:
            candidates.append(item)
            data.pop(i)
    candidates_num = len(candidates)
    parts = [[] for _ in range(num)]
    part_id = 0
    while candidates:
        parts[part_id].append(candidates.pop())
        part_id += 1
        if part_id == num:
            part_id = 0

    #Delete all intermediate files before re inference
    for path in json_files_paths:
        try:
            os.remove(path)
        except Exception as e:
            pass

    manager = multiprocessing.Manager()
    progress_dict = manager.dict()

    # init progress dict
    for i in range(len(parts)):
        progress_dict[i] = 0
    #Collect relevant contexts with multiple processes
    processes = []
    #Multiprocess
    for i, part in enumerate(parts):
        path = json_files_paths[i]
        process = multiprocessing.Process(target=subprocessing_module, args=(i,part,path,data_provider,task,task_num,progress_dict,))
        processes.append(process)
        process.start()

    with tqdm(total=candidates_num) as pbar:
        previous_progress = {i: 0 for i in range(len(parts))}
        while any(process.is_alive() for process in processes):
            current_progress = sum(progress_dict.values())
            pbar.update(current_progress - sum(previous_progress.values()))
            previous_progress = progress_dict.copy()
            time.sleep(1)

        while pbar.n < candidates_num:
            current_progress = sum(progress_dict.values())
            pbar.update(candidates_num - pbar.n)
            time.sleep(1)

    for process in processes:
        process.join()

    pbar.close()
    for process in processes:
        process.join()
    print("All workers for re-inference have finished")

    #Merge json files into one file
    json_data_merge(dataset_name, re_inference_path, json_files_paths)

    with open(re_inference_path, 'r', encoding='utf-8') as file:
        new_data = json.load(file)
    for item in new_data:
        data.append(item)
    
    #Add re_infered info into main processed file
    with open(merged_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    candidates = []
    for i, item in enumerate(data):
        term_info = item["term_info"] 
        if len(term_info) < 50:
            candidates.append(item)
            data.pop(i)
    candidates_num = len(candidates)
    return candidates_num






