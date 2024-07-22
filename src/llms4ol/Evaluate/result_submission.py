from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model,PeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification,T5Tokenizer,T5ForConditionalGeneration
from llms4ol.Evaluate.evaluate_metrics import EvaluationMetrics
from llms4ol.DataProcess.Dataset_Builder import GeoNames_DataBuilder,GO_DataBuilder,Schema_DataBuilder,UMLS_DataBuilder,WordNet_DataBuilder
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from collections import defaultdict
import torch,random,json
from llms4ol.path import *
import torch.nn.functional as F

def type_extraction(filename):
    id2label = defaultdict(str)
    label2id = defaultdict(int)

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            label = line.strip()
            if label: 
                id2label[idx] = label
                label2id[label] = idx
    return id2label,label2id

def taskA_evaluator(task_num,subtask_num):
    a_1_1 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/A.1(FS)_WordNet_Test.json"
    a_2_1 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/A.2(FS)_GeoNames_Test.json"
    a_3_1 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/A.3(FS)_UMLS_MEDCIN_Test.json"
    a_3_2 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/A.3(FS)_UMLS_NCI_Test.json"
    a_3_3 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/A.3(FS)_UMLS_SNOMED_Test.json"
    a_4_1 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/A.4(FS)_GO_Biological_Process_Test.json"
    a_4_2 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/A.4(FS)_GO_Cellular_Component_Test.json"
    a_4_3 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/A.4(FS)_GO_Molecular_Function_Test.json"
    a_5_1 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/other/A.5(ZS)_DBpedia_Test.json"
    a_6_1 = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/other/A.6(ZS)_FoodOn_Test.json"

    task2kb = {1:"WordNet",2:"GeoNames",3:"UMLS",4:"GO"}
    kb_name = task2kb[task_num]
    val = f"a_{task_num}_{subtask_num}"

    test_file_path = locals()[val]
    with open(test_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if task_num in [1,2]:
        path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/{kb_name}/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
        model_path = find_trained_model_path(path_pattern)
        if task_num == 1:
            id2label,label2id,_,_,_ = WordNet_DataBuilder.WN_TaskA_TextClf_dataset_builder()
        else:
            id2label,label2id,_,_ = GeoNames_DataBuilder.Geo_TaskA_TextClf_dataset_builder()
    elif task_num in [3,4]:
        path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/{kb_name}/Finetune/{subtask_num}_llama3_Textclf_with_Context_full/checkpoint-*"
        model_path = find_trained_model_path(path_pattern)
        if task_num == 3:
            if subtask_num == 1:
                id2label,label2id,_,_ = UMLS_DataBuilder.UMLS_MT_TaskA_TextClf_dataset_builder()
            elif subtask_num == 2:
                id2label,label2id,_,_ = UMLS_DataBuilder.UMLS_NT_TaskA_TextClf_dataset_builder()
            elif subtask_num == 3:
                id2label,label2id,_,_ = UMLS_DataBuilder.UMLS_SU_TaskA_TextClf_dataset_builder()
        else:
            if subtask_num == 1:
                id2label,label2id,_,_ = GO_DataBuilder.GO_BP_TaskA_TextClf_dataset_builder()
            elif subtask_num == 2:
                id2label,label2id,_,_ = GO_DataBuilder.GO_CC_TaskA_TextClf_dataset_builder()
            elif subtask_num == 3:
                id2label,label2id,_,_ = GO_DataBuilder.GO_MF_TaskA_TextClf_dataset_builder()
    else:
        model_path = "meta-llama/Meta-Llama-3-8B"
        if task_num == 5:
            file_path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/other/A.5(ZS)-DBpedia-Types.txt"
        else:
            file_path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/test_datasets/other/A.6(ZS)-FoodOn-Types.txt"
        id2label,label2id = type_extraction(file_path)


    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        label2id=label2id,
        id2label=id2label,
        device_map="cuda"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    predict_data = []
    predicted_label = []
    for item in data:
        id = item["ID"]
        temp = item["term"]
        if task_num == 1:#WordNet
            sentence = item["sentence"]
            single_text = f"{temp} [SEP] {sentence}"
        elif task_num == 2: #GeoNames
            single_text = f"What is the geographical type of {temp}?"
        else:
            single_text = temp
        inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item() #top 1
            # 使用 softmax 将 logits 转换为概率
        #probabilities = F.softmax(logits, dim=-1).cpu().numpy()

        # 获取所有类别及其对应的概率
        #all_classes_probabilities = {i: prob for i, prob in enumerate(probabilities[0])}
        
        # 输出所有类别和对应的概率
        #print(f"ID: {id}, Term: {temp}, predict_type: {id2label[predicted_class_id]}, Probabilities: {all_classes_probabilities}")
        predicted_label.append(predicted_class_id)
        predict_data.append({
            "ID": id,
            "type": [id2label[predicted_class_id]]
        })
    root_path = find_root_path()
    file_path = root_path + f"/src/llms4ol/Evaluate/submission_files/{subtask_num}TaskA_{kb_name}_predict_result.json"
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(predict_data, file, ensure_ascii=False, indent=4)

for tn in [3,4]:
    for sn in [1,2,3]:
        taskA_evaluator(tn,sn)