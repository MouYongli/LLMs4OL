from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model,PeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification,T5Tokenizer,T5ForConditionalGeneration
from llms4ol.Evaluate.evaluate_metrics import EvaluationMetrics
from llms4ol.DataProcess.Dataset_Builder import GeoNames_DataBuilder,GO_DataBuilder,Schema_DataBuilder,UMLS_DataBuilder,WordNet_DataBuilder
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import torch,random,json
from llms4ol.path import *

def taskA_aacuracy(kb_name):
    root_path = find_root_path()

    def load_json(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    data1 = load_json(root_path + f"/src/llms4ol/Evaluate/submission_files/TaskA_{kb_name}_predict_result.json")
    data2 = load_json(root_path + f"/src/llms4ol/Evaluate/submission_files/TaskA_{kb_name}_actual_result.json")
    # extract into dictionary
    def create_term_dict(data):
        term_dict = {}
        for item in data:
            term = item['term']
            types = set(item['type'])
            term_dict[term] = types
        return term_dict

    term_dict1 = create_term_dict(data1)
    term_dict2 = create_term_dict(data2)

    # compare the type of the term exists in both files
    terms = set(term_dict1.keys()).intersection(set(term_dict2.keys()))
    correct_predictions = 0
    total_predictions = len(terms)
    for term in terms:
        if term_dict1[term] == term_dict2[term]:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"Accuracy: {accuracy}")

def taskA_evaluater(model_name,kb_name):
    root_path = find_root_path()
    def random_sample_corresponding(list1, list2, list3, sample_size):
        if len(list3) == 0:
            combined_list = list(zip(list1, list2))
            # Step 2: Perform random sampling on the combined list
            sampled_combined_list = random.sample(combined_list, sample_size)
            
            # Step 3: Unzip the sampled list of tuples back into lists
            sampled_list1, sampled_list2 = zip(*sampled_combined_list)
            
            return list(sampled_list1), list(sampled_list2)
        else:
            combined_list = list(zip(list1, list2, list3))

            # Step 2: Perform random sampling on the combined list
            sampled_combined_list = random.sample(combined_list, sample_size)
            
            # Step 3: Unzip the sampled list of tuples back into lists
            sampled_list1, sampled_list2, sampled_list3 = zip(*sampled_combined_list)
            
            return list(sampled_list1), list(sampled_list2), list(sampled_list3)

    if model_name == "roberta":
        if kb_name == "geonames":
            id2label,label2id,text,label = GeoNames_DataBuilder.Geo_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Finetune/roberta_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "wordnet":
            id2label,label2id,text,label,sentences = WordNet_DataBuilder.WN_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label,sentences = random_sample_corresponding(text, label, sentences, int(0.2*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/roberta_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "umls":
            id2label,label2id,text,label = UMLS_DataBuilder.UMLS_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Finetune/roberta_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "go":
            id2label,label2id,text,label = GO_DataBuilder.GO_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GO/Finetune/roberta_Textclf_with_Context_full/checkpoint-*"
        model_path = find_trained_model_path(path_pattern)
    elif model_name == "llama3":
        if kb_name == "geonames":
            id2label,label2id,text,label = GeoNames_DataBuilder.Geo_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "wordnet":
            id2label,label2id,text,label,sentences = WordNet_DataBuilder.WN_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label,sentences = random_sample_corresponding(text, label, sentences, int(0.2*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "umls":
            id2label,label2id,text,label = UMLS_DataBuilder.UMLS_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "go":
            id2label,label2id,text,label = GO_DataBuilder.GO_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GO/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
        model_path = find_trained_model_path(path_pattern)
    elif model_name == "t5":
        if kb_name == "geonames":
            id2label,label2id,text,label = GeoNames_DataBuilder.Geo_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Finetune/flan-t5-xl_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "wordnet":
            id2label,label2id,text,label,sentences = WordNet_DataBuilder.WN_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label,sentences = random_sample_corresponding(text, label, sentences, int(0.2*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/flan-t5-xl_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "umls":
            id2label,label2id,text,label = UMLS_DataBuilder.UMLS_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Finetune/flan-t5-xl_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "go":
            id2label,label2id,text,label = GO_DataBuilder.GO_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GO/Finetune/flan-t5-xl_Textclf_with_Context_full/checkpoint-*"
        model_path = find_trained_model_path(path_pattern)


    test_text = text
    test_label = label
    dataset = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text})})

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        label2id=label2id,
        id2label=id2label,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    predict_data = []
    actual_data = []
    actual_label = []
    predicted_label = []
    for item in dataset["test"]:
        a_label = item['label']
        actual_label.append(item['label'])
        temp = item['text']
        if kb_name == "geonames":
            single_text = f"What is the geographical type of {temp}?"
        elif kb_name == "wordnet":
            single_text = f"{temp}"
        elif kb_name == "umls":
            single_text = f"{temp}"
        elif kb_name == "go":
            single_text = f"{temp}"
        inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item() #top 1
            predicted_label.append(predicted_class_id)
            predict_data.append({
                "term": temp,
                "type": [id2label[predicted_class_id]]
            })
            actual_data.append({
                "term": temp,
                "type": [id2label[a_label]]
            })

    results = EvaluationMetrics.evaluate_accuracy(actual=actual_label, predicted=predicted_label)
    print(results['accuracy'])
    file_path = root_path + f"/src/llms4ol/Evaluate/submission_files/TaskA_{kb_name}_predict_result.json"
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(predict_data, file, ensure_ascii=False, indent=4)

    file_path = root_path + f"/src/llms4ol/Evaluate/submission_files/TaskA_{kb_name}_actual_result.json"
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(actual_data, file, ensure_ascii=False, indent=4)
    #taskA_aacuracy(kb_name)


#taskA_evaluater("llama3","wordnet")

def evaluator_for_WN():
    result_file = "/home/yxpeng/Projects/LLMs4OL/src/llms4ol/Evaluate/submission_files/TaskA_WN_Results.json"
    test_file = "/home/yxpeng/Projects/LLMs4OL/src/llms4ol/Evaluate/target_files/WordNet_Eva.json"
    with open(result_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    with open(test_file, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    print(len(data))
    print(len(test_data))
    ids_done = {item["ID"]: item for item in data}
    ids_test = {item["ID"]: item for item in test_data}
    total_predicted_num = len(data)
    total_actual_num = len(test_data)
    correct_num = 0
    for item in test_data:
        if ids_done[item["ID"]]["type"][0] == ids_test[item["ID"]]["type"]:
            correct_num += 1
    print(correct_num)
    print(correct_num / total_predicted_num)
evaluator_for_WN()