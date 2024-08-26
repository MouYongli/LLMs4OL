from peft import LoraConfig, get_peft_model,PeftModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer,AutoModelForSequenceClassification,EarlyStoppingCallback,AutoConfig,AutoModelForCausalLM,DataCollatorForLanguageModeling
from llms4ol.DataProcess.Dataset_Builder import GeoNames_DataBuilder,GO_DataBuilder,Schema_DataBuilder,UMLS_DataBuilder,WordNet_DataBuilder
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from llms4ol.path import find_root_path
import evaluate,json
import numpy as np
import torch

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}. All model parameters: {all_model_params} ")
    return trainable_model_params

def taskA_Pretrain_GeoNames():
    #train as causal LM
    #model_id = "meta-llama/Meta-Llama-3-8B"
    #model_id = "/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Pretrain/llama3_pretrain_full/checkpoint-674"
    #model_id = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Pretrain/llama3_pretrain_final_full/checkpoint-625"
    model_id = "/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Pretrain/llama3_pretrain_final_3/checkpoint-900" #需要被替换
    output_dir="/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Pretrain/llama3_pretrain_final_4"

    text = GeoNames_DataBuilder.Geo_Pretrain_dataset_builder()

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    dataset = DatasetDict({'train': Dataset.from_dict({'text': text})})

    print(dataset)

    #Preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = dataset.map(preprocess_function, batched=True,remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #Train
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
    model.config.pad_token_id = model.config.eos_token_id
    ori_p = print_number_of_trainable_model_parameters(model)

    batch_size = 2
    epochs = 1
    steps = 16

    # Without validation 
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=200,
        disable_tqdm=False,
        logging_steps=10,
        save_total_limit=1,
        gradient_accumulation_steps=steps,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

def taskA_Pretrain(kb_name):
    #train as causal LM
    #model_id = "meta-llama/Meta-Llama-3-8B"
    if kb_name == "go":
        model_id = "meta-llama/Meta-Llama-3-8B"
        output_dir="/home/yxpeng/DATA/Checkpoints/TaskA/GO/Pretrain/llama3_pretrain_"
        path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.4-GO/GO_Types_processed.json"
    elif kb_name == "geonames":
        model_id = "/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Pretrain/llama3_pretrain_full/checkpoint-674"
        output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Pretrain/llama3_pretrain_"
        path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.1-GeoNames/geoTypes_processed.json"
    elif kb_name == "umls":
        model_id = "/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Pretrain/llama3_pretrain_full/checkpoint-76"
        output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Pretrain/llama3_pretrain_"
        path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.3-UMLS/umls_Types_processed.json"
    elif kb_name == "schema":
        model_id = "/home/yxpeng/DATA/Checkpoints/TaskA/Schema/Pretrain/llama3_pretrain_full/checkpoint-1054"
        output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Pretrain/llama3_pretrain_"
        path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.2-Schema.org/schemaTypes_processed.json"

    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    text = []
    for item in data:
        item_type = item["term"]
        item_info = item["term_info"]
        text.append(f"Type:'{item_type}', {item_info}")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    dataset = DatasetDict({'train': Dataset.from_dict({'text': text})})

    print(dataset)

    #Preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = dataset.map(preprocess_function, batched=True,remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #Train
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.config.pad_token_id = model.config.eos_token_id
    ori_p = print_number_of_trainable_model_parameters(model)

    batch_size = 2
    epochs = 2
    output_dir += "full"
    steps = 1
    if kb_name == "go":
        steps = 4

    # Without validation 
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        disable_tqdm=False,
        logging_steps=10,
        save_total_limit=1,
        gradient_accumulation_steps=steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()


def taskA_trainer(model_name,kb_name,finetune_methode,trained_model_path=None, num_subtask=1):
    if kb_name == "geonames":
        id2label,label2id,text,label = GeoNames_DataBuilder.Geo_TaskA_TextClf_dataset_builder()
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Finetune/roberta_Textclf_with_Context_"
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Finetune/llama3_Textclf_with_Context_"
        elif model_name == "t5":
            model_id = "google/flan-t5-xl"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Finetune/flan-t5-xl_Textclf_with_Context_"
    elif kb_name == "wordnet":
        id2label,label2id,text,label,sentences = WordNet_DataBuilder.WN_TaskA_TextClf_dataset_builder()
        for i, item in enumerate(text):
            item +=  " [SEP] "
            item +=  sentences[i]
            text[i] = item
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/roberta_Textclf_with_Context_"
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/llama3_Textclf_with_Context_"
        elif model_name == "t5":
            model_id = "google/flan-t5-xl"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/flan-t5-xl_Textclf_with_Context_"
    elif kb_name == "umls":
        if num_subtask == 1:
            id2label,label2id,text,label = UMLS_DataBuilder.UMLS_MT_TaskA_TextClf_dataset_builder()
        elif num_subtask == 2:
            id2label,label2id,text,label = UMLS_DataBuilder.UMLS_NT_TaskA_TextClf_dataset_builder()
        elif num_subtask == 3:
            id2label,label2id,text,label = UMLS_DataBuilder.UMLS_SU_TaskA_TextClf_dataset_builder()
        
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir=f"/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Finetune/{num_subtask}_roberta_Textclf_with_Context_"
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir=f"/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Finetune/{num_subtask}_llama3_Textclf_with_Context_"
        elif model_name == "t5":
            model_id = "google/flan-t5-xl"
            output_dir=f"/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Finetune/{num_subtask}_flan-t5-xl_Textclf_with_Context_"
    elif kb_name == "go":
        if num_subtask == 1:
            id2label,label2id,text,label = GO_DataBuilder.GO_BP_TaskA_TextClf_dataset_builder()
        elif num_subtask == 2:
            id2label,label2id,text,label = GO_DataBuilder.GO_CC_TaskA_TextClf_dataset_builder()
        elif num_subtask == 3:
            id2label,label2id,text,label = GO_DataBuilder.GO_MF_TaskA_TextClf_dataset_builder()

        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir=f"/home/yxpeng/DATA/Checkpoints/TaskA/GO/Finetune/{num_subtask}_roberta_Textclf_with_Context_"
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir=f"/home/yxpeng/DATA/Checkpoints/TaskA/GO/Finetune/{num_subtask}_llama3_Textclf_with_Context_"
        elif model_name == "t5":
            model_id = "google/flan-t5-xl"
            output_dir=f"/home/yxpeng/DATA/Checkpoints/TaskA/GO/Finetune/{num_subtask}_flan-t5-xl_Textclf_with_Context_"

    def preprocess_function(examples):
        return tokenizer(examples["term"], truncation=True)
    
    dataset = DatasetDict({'train': Dataset.from_dict({'label': label, 'term': text})})

    print(dataset)

    #continue to train the trained model
    if trained_model_path:
        model_id = trained_model_path
    #Preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model_name != "t5":
        tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    #Train
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(id2label), id2label=id2label, label2id=label2id, device_map="auto"
    )
    model.config.pad_token_id = model.config.eos_token_id
    ori_p = print_number_of_trainable_model_parameters(model)

    if model_name == "roberta":
        batch_size = 200
    else:
        if kb_name == "umls":
            batch_size = 150
        else:
            batch_size = 150
    if model_name == "roberta":
        epochs = 2
    else:
        epochs = 2
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
    elif finetune_methode == "full":
        output_dir += "full"

    # Without validation 
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        disable_tqdm=False,
        logging_steps=10,
        save_total_limit=1,
        fp16=True,
        gradient_accumulation_steps=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

def taskB_trainer(model_name,kb_name,finetune_methode, dataset_build_method, trained_model_path=None):

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    accuracy = evaluate.load("accuracy")
    root_path = find_root_path()

    if kb_name == "geonames":
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.1-GeoNames/geoname_train_pairs.json"

        if dataset_build_method == 1:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = GeoNames_DataBuilder.m1_Geo_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 2:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = GeoNames_DataBuilder.m2_Geo_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 3:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = GeoNames_DataBuilder.m3_Geo_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 4:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = GeoNames_DataBuilder.m4_Geo_TaskB_TextClf_train_dataset_builder(train_dataset_path)

        eval_label, eval_text, eval_context = GeoNames_DataBuilder.Geo_TaskB_TextClf_evl_dataset_builder(train_dataset_path)
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/roberta_Textclf_with_Context_" + method_name
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/llama3_Textclf_with_Context_"+ method_name
            if finetune_methode == "continue":
                model_id = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Pretrain/llama3_pretrain_full/checkpoint-674"
                output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Pretrain_Finetune/llama3_Pretrained_Textclf_with_Context_"+ method_name
        elif model_name == "t5":
            model_id = "google/flan-t5-xl"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/flan-t5-xl_Textclf_with_Context_"+ method_name
    elif kb_name == "schema":
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.2-Schema.org/schemaorg_train_pairs.json"
        if dataset_build_method == 1:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = Schema_DataBuilder.m1_Schema_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 2:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = Schema_DataBuilder.m2_Schema_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 3:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = Schema_DataBuilder.m3_Schema_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 4:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = Schema_DataBuilder.m4_Schema_TaskB_TextClf_train_dataset_builder(train_dataset_path)

        eval_label, eval_text, eval_context = Schema_DataBuilder.Schema_TaskB_TextClf_evl_dataset_builder(train_dataset_path)
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/roberta_Textclf_with_Context_"+ method_name
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/llama3_Textclf_with_Context_"+ method_name
            if finetune_methode == "continue":
                model_id = "/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Pretrain_Finetune/llama3_Pretrained_Textclf_with_Context_m2_/checkpoint-1556"
                output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Pretrain_Finetune/llama3_Pretrained_Textclf_with_Context_update_"+ method_name
        elif model_name == "t5":
            model_id = "google/flan-t5-xl"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/flan-t5-xl_Textclf_with_Context_"+ method_name
    elif kb_name == "umls":
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.3-UMLS/umls_train_pairs.json"
        if dataset_build_method == 1:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = UMLS_DataBuilder.m1_UMLS_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 2:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = UMLS_DataBuilder.m2_UMLS_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 3:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = UMLS_DataBuilder.m3_UMLS_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 4:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = UMLS_DataBuilder.m4_UMLS_TaskB_TextClf_train_dataset_builder(train_dataset_path)

        eval_label, eval_text, eval_context = UMLS_DataBuilder.UMLS_TaskB_TextClf_evl_dataset_builder(train_dataset_path)
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/roberta_Textclf_with_Context_"+ method_name
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/llama3_Textclf_with_Context_"+ method_name
            if finetune_methode == "continue":
                model_id = "/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Pretrain/llama3_pretrain_full/checkpoint-76"
                output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Pretrain_Finetune/llama3_Pretrained_Textclf_with_Context_"+ method_name
        elif model_name == "t5":
            model_id = "google/flan-t5-xl"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/flan-t5-xl_Textclf_with_Context_"+ method_name
    elif kb_name == "go":
        train_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.4-GO/go_train_pairs.json"
        if dataset_build_method == 1:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = GO_DataBuilder.m1_GO_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 2:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = GO_DataBuilder.m2_GO_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 3:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = GO_DataBuilder.m3_GO_TaskB_TextClf_train_dataset_builder(train_dataset_path)
        elif dataset_build_method == 4:
            method_name = "m" + str(dataset_build_method) + "_"
            label, text, context = GO_DataBuilder.m4_GO_TaskB_TextClf_train_dataset_builder(train_dataset_path)

        eval_label, eval_text, eval_context = GO_DataBuilder.GO_TaskB_TextClf_evl_dataset_builder(train_dataset_path)
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/roberta_Textclf_with_Context_"+ method_name
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/llama3_Textclf_with_Context_"+ method_name
        elif model_name == "t5":
            model_id = "google/flan-t5-xl"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/flan-t5-xl_Textclf_with_Context_"+ method_name


    def preprocess_function(examples):
        inputs = [f"{text} ## {context}" for context, text in zip(examples["context"], examples["text"])]
        tokenized_input = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        
        return tokenized_input

    dataset = DatasetDict({'train': Dataset.from_dict({'label': label, 'text': text, 'context': context}),
                            'eval': Dataset.from_dict({'label': eval_label, 'text': eval_text, 'context': eval_context})
                            })
    # mianly for testing m2 -> m3 OR m3 -> m2
    if trained_model_path:
        model_id = trained_model_path
        output_dir += "_continue_"

    print(dataset)

    #Preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model_name != "t5":
        tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    #Train
    id2label = {0: "incorrect", 1: "correct"}
    label2id = {"incorrect": 0, "correct": 1}


    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, id2label=id2label, label2id=label2id, device_map="auto"
    )
    model.config.pad_token_id = model.config.eos_token_id
    ori_p = print_number_of_trainable_model_parameters(model)

    if model_name == "roberta":
        batch_size = 64
    else:
        batch_size = 8
        steps = 8

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
    elif finetune_methode == "full":
        output_dir += "full"

    # without validation 
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        disable_tqdm=False,
        logging_steps=10,
        save_total_limit=1,
        gradient_accumulation_steps=steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    #cd ~/Projects/LLMs4OL/src/llms4ol/Training/flan_t5_xl
    #CUDA_VISIBLE_DEVICES=0,1,2,3 python ./main.py
    trainer.train()

