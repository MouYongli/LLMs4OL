from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer,AutoModelForSequenceClassification,EarlyStoppingCallback,AutoConfig,AutoModelForCausalLM
from llms4ol.DataProcess.Dataset_Builder import GeoNames_DataBuilder,GO_DataBuilder,Schema_DataBuilder,UMLS_DataBuilder,WordNet_DataBuilder
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from llms4ol.path import find_root_path
import evaluate
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

def taskA_trainer(model_name,kb_name,finetune_methode, dataset_build_method):
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

        eval_label, eval_text = GeoNames_DataBuilder.Geo_TaskB_TextClf_evl_dataset_builder(train_dataset_path)
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/roberta_Textclf_with_Context_" + method_name
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/llama3_Textclf_with_Context_"+ method_name
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

        eval_label, eval_text = Schema_DataBuilder.Schema_TaskB_TextClf_evl_dataset_builder(train_dataset_path)
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/roberta_Textclf_with_Context_"+ method_name
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/llama3_Textclf_with_Context_"+ method_name
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

        eval_label, eval_text = UMLS_DataBuilder.UMLS_TaskB_TextClf_evl_dataset_builder(train_dataset_path)
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/roberta_Textclf_with_Context_"+ method_name
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/llama3_Textclf_with_Context_"+ method_name
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

        eval_label, eval_text = GO_DataBuilder.GO_TaskB_TextClf_evl_dataset_builder(train_dataset_path)
        if model_name == "roberta":
            model_id = "FacebookAI/roberta-large"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/roberta_Textclf_with_Context_"+ method_name
        elif model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/llama3_Textclf_with_Context_"+ method_name
        elif model_name == "t5":
            model_id = "google/flan-t5-xl"
            output_dir="/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/flan-t5-xl_Textclf_with_Context_"+ method_name

    # load model from huggingface
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_cache=False,
        device_map="auto"
    )
    # load tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    ori_p = print_number_of_trainable_model_parameters(model)

    if model_name == "roberta":
        batch_size = 64
    else:
        batch_size = 16

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
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        peft_p = print_number_of_trainable_model_parameters(model)
        print(f"# Trainable Parameter \nBefore: {ori_p} \nAfter: {peft_p} \nPercentage: {round(peft_p / ori_p * 100, 2)}")
        output_dir += "lora"
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            weight_decay=0.01,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            disable_tqdm=False,
            logging_steps=10,
            save_total_limit=1,
        )
    elif finetune_methode == "full":
        output_dir += "full"
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            weight_decay=0.01,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            disable_tqdm=False,
            logging_steps=10,
            save_total_limit=1,
        )
    if len(context) == 0:
        dataset = DatasetDict({'train': Dataset.from_dict({'label': label, 'text': text}),
                               'eval': Dataset.from_dict({'label': eval_label, 'text': eval_text})
                               })
    else:
        dataset = DatasetDict({'train': Dataset.from_dict({'label': label, 'text': text, 'context': context}),
                               'eval': Dataset.from_dict({'label': eval_label, 'text': eval_text})
                               })

    print(dataset)

    ### generate prompt based on template ###
    prompt_template = {
        "prompt_input": \
            "Below is an instruction that describes a task, paired with an input that provides further context.\
            Write a response that appropriately completes the request.\
            \n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",

        "prompt_no_input": \
            "Below is an instruction that describes a task.\
            Write a response that appropriately completes the request.\
            \n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:"
    }

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
        inputs = [f"{text} ### {context}" for context, text in zip(examples["context"], examples["text"])]
        max_length = 512
        stride = 256 
        tokenized_inputs = {'input_ids': [], 'attention_mask': []}
        
        for input in inputs:
            tokenized_input = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
            input_ids = tokenized_input['input_ids'][0]
            attention_mask = tokenized_input['attention_mask'][0]
            start = 0
            while start < len(input_ids):
                end = min(start + max_length, len(input_ids))
                chunk = input_ids[start:end]
                chunk_attention_mask = attention_mask[start:end]
                # the length of each chunk is max_lengty
                if len(chunk) < max_length:
                    padding_length = max_length - len(chunk)
                    chunk = torch.cat([chunk, torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)])
                    chunk_attention_mask = torch.cat([chunk_attention_mask, torch.zeros((padding_length,), dtype=torch.long)])
                tokenized_inputs['input_ids'].append(chunk)
                tokenized_inputs['attention_mask'].append(chunk_attention_mask)
                if end == len(input_ids):
                    break
                start += stride

        tokenized_inputs['input_ids'] = torch.stack(tokenized_inputs['input_ids'])
        tokenized_inputs['attention_mask'] = torch.stack(tokenized_inputs['attention_mask'])
        
        return tokenized_inputs

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
        batch_size = 12

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
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        disable_tqdm=False,
        logging_steps=10,
        save_total_limit=1,
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

