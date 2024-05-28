from transformers import pipeline
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model,PeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification
from llms4ol.Evaluate.evaluate_metrics import EvaluationMetrics
from llms4ol.DataProcess.GeoNames.GeoNames_DataBuilder import *
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import torch
from llms4ol.path import *

def taskB_evaluater(model_name,kb_name,finetune_methode,peft_path=None,trained_model_path = None):

    if model_name == "roberta":
        model_path = "FacebookAI/roberta-large"
    elif model_name == "llama3":
        model_path = "meta-llama/Meta-Llama-3-8B"
    elif model_name == "t5":
        model_path = "google/flan-t5-xl"

    if finetune_methode == "lora" and peft_path:
        peft_path = peft_path
    elif finetune_methode == "full" and trained_model_path:
        model_path = trained_model_path
    elif finetune_methode == "lora" and peft_path == None:
        if model_name == "roberta":
            path_pattern = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/roberta_Context_lora/checkpoint-*"
            peft_path = find_trained_model_path(path_pattern)
        elif model_name == "llama3":
            path_pattern = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/llama3_Context_lora/checkpoint-*"
            peft_path = find_trained_model_path(path_pattern)
        elif model_name == "t5":
            path_pattern = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/flan-t5-xl_Context_lora/checkpoint-*"
            peft_path = find_trained_model_path(path_pattern)
    elif finetune_methode == "full" and trained_model_path == None:
        if model_name == "roberta":
            path_pattern = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/roberta_Context_Full/checkpoint-*"
            model_path = find_trained_model_path(path_pattern)
        elif model_name == "llama3":
            path_pattern = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/llama3_Context_Full/checkpoint-*"
            model_path = find_trained_model_path(path_pattern)
        elif model_name == "t5":
            path_pattern = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/flan-t5-xl_Context_Full/checkpoint-*"
            model_path = find_trained_model_path(path_pattern)


    root_path = find_root_path()
    if kb_name == "geonames":
        test_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.1-GeoNames/test_dataset/hierarchy_test.json"
    elif kb_name == "schema":
        test_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.2-Schema.org/schemaorg_train_pairs.json"
    elif kb_name == "umls":
        test_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.3-UMLS/umls_train_pairs.json"
    elif kb_name == "go":
        test_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.4-GO/go_train_pairs.json"

    eval_label, eval_text = FintuneB_evl_dataset_builder(test_dataset_path)
    dataset = DatasetDict({'eval': Dataset.from_dict({'label': eval_label, 'text': eval_text})})

    id2label = {0: "incorrect", 1: "correct"}
    label2id = {"incorrect": 0, "correct": 1}


    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        label2id=label2id,
        id2label=id2label,
        device_map="cuda"
    )
    print(model_path, peft_path)
    if finetune_methode =="lora":
        # loading peft weight
        model = PeftModelForSequenceClassification.from_pretrained(
            model,
            peft_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    actual_label = []
    predicted_label = []

    for item in dataset["eval"]:
        actual_label.append(item['label'])
        single_text = item['text']
        inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            predicted_label.append(predicted_class_id)

    results = EvaluationMetrics.evaluate(actual=actual_label, predicted=predicted_label)
    print(f"{model_name} with finetune methode {finetune_methode} has :")
    print("F1-score:", results['clf-report-dict']['macro avg']['f1-score'])
