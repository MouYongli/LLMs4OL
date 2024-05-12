from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModel, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer,AutoModelForSequenceClassification
from llms4ol.DataProcess.simple_ds_preprocess import *
import evaluate
import numpy as np
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import torch


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model_id = "google/flan-t5-xl"
train_dataset_path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.1-GeoNames/geoname_train_pairs.json"
eval_dataset_path = "/home/yxpeng/Projects/LLMs4OL/src/assets/Datasets/SubTaskB.1-GeoNames/test_dataset/hierarchy_test.json"
label, text, context = train_data_handler(train_dataset_path)
eval_label, eval_text = eval_data_handler(eval_dataset_path)
dataset = DatasetDict({'train': Dataset.from_dict({'label': label, 'text': text, 'context': context}),
                       'eval': Dataset.from_dict({'label': eval_label, 'text': eval_text})})
print(dataset)

#Preproce
tokenizer = AutoTokenizer.from_pretrained(model_id)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_geo = dataset.map(preprocess_function, batched=True)

#Evaluate
accuracy = evaluate.load("accuracy")

#Train
id2label = {0: "incorrect", 1: "correct"}
label2id = {"incorrect": 0, "correct": 1}


model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2, id2label=id2label, label2id=label2id, device_map="auto"
)

ori_p = print_number_of_trainable_model_parameters(model)
print(ori_p)
# model = prepare_model_for_kbit_training(model)

# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=None,  # peft will figure out the correct target_modules based on the model_type
#     bias="none",
#     task_type="SEQ_CLS",
# )
# #Task_Type
# # SEQ_CLS: Text classification.
# # SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
# # CAUSAL_LM: Causal language modeling.
# # TOKEN_CLS: Token classification.
# # QUESTION_ANS: Question answering.
# # FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features for downstream tasks.



# model = get_peft_model(model, peft_config)
# peft_p = print_number_of_trainable_model_parameters(model)
# print(f"# Trainable Parameter \nBefore: {ori_p} \nAfter: {peft_p} \nPercentage: {round(peft_p / ori_p * 100, 2)}")

training_args = TrainingArguments(
    output_dir="/home/yxpeng/DATA/Tuning/flan-t5-xl_finetuned_model",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    disable_tqdm=False,
    logging_steps=10,
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_geo["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

#cd ~/Projects/LLMs4OL/src/llms4ol/Training/flan_t5_xl
#CUDA_VISIBLE_DEVICES=0,1,2,3 python ./main.py
trainer.train()

