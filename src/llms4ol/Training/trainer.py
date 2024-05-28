from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer,AutoModelForSequenceClassification
from llms4ol.DataProcess.GeoNames.GeoNames_DataBuilder import *
import evaluate
import numpy as np
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import torch
from llms4ol.path import find_root_path

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}. All model parameters: {all_model_params} ")
    return trainable_model_params

def taskB_trainer(model_name,kb_name,finetune_methode):

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    if model_name == "roberta":
        model_id = "FacebookAI/roberta-large"
        output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/roberta_Context_"
    elif model_name == "llama3":
        model_id = "meta-llama/Meta-Llama-3-8B"
        output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/llama3_Context_"
    elif model_name == "t5":
        model_id = "google/flan-t5-xl"
        output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/flan-t5-xl_Context_"

    root_path = find_root_path()
    if kb_name == "geonames":
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.1-GeoNames/geoname_train_pairs.json"
        label, text, context = FintuneB_dataset_builder(train_dataset_path)
    elif kb_name == "schema":
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.2-Schema.org/schemaorg_train_pairs.json"
    elif kb_name == "umls":
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.3-UMLS/umls_train_pairs.json"
    elif kb_name == "go":
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.4-GO/go_train_pairs.json"
    
    
    dataset = DatasetDict({'train': Dataset.from_dict({'label': label, 'text': text, 'context': context})
                        })
    print(dataset)

    #Preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model_name != "t5":
        tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_geo = dataset.map(preprocess_function, batched=True)

    #Train
    id2label = {0: "incorrect", 1: "correct"}
    label2id = {"incorrect": 0, "correct": 1}


    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, id2label=id2label, label2id=label2id, device_map="auto"
    )
    model.config.pad_token_id = model.config.eos_token_id
    ori_p = print_number_of_trainable_model_parameters(model)
    print(ori_p)

    if finetune_methode == "lora":
        #Task_Type
        # SEQ_CLS: Text classification.
        # SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
        # CAUSAL_LM: Causal language modeling.
        # TOKEN_CLS: Token classification.
        # QUESTION_ANS: Question answering.
        # FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features for downstream tasks.
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=None,  # peft will figure out the correct target_modules based on the model_type
            task_type="SEQ_CLS",
        )

        model = get_peft_model(model, peft_config)
        peft_p = print_number_of_trainable_model_parameters(model)
        print(f"# Trainable Parameter \nBefore: {ori_p} \nAfter: {peft_p} \nPercentage: {round(peft_p / ori_p * 100, 2)}")
        output_dir += "lora"
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            num_train_epochs=20,
            weight_decay=0.01,
            save_strategy="epoch",
            disable_tqdm=False,
            logging_steps=10,
            save_total_limit=1,
        )
    elif finetune_methode == "full":
        output_dir += "full"
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            num_train_epochs=10,
            weight_decay=0.01,
            save_strategy="epoch",
            disable_tqdm=False,
            logging_steps=10,
            save_total_limit=1,
        )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_geo["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    #cd ~/Projects/LLMs4OL/src/llms4ol/Training/flan_t5_xl
    #CUDA_VISIBLE_DEVICES=0,1,2,3 python ./main.py
    trainer.train()

