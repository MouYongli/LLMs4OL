#Load Dataset
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from llms4ol.Training.Useful_Function_For_Training import print_number_of_trainable_model_parameters
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

imdb = load_dataset("imdb")
print(imdb)


#Preprocess
tokenizer = AutoTokenizer.from_pretrained("../../assets/LLMs/flan-t5-base")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_imdb = imdb.map(preprocess_function, batched=True)

#Evaluate
accuracy = evaluate.load("accuracy")

#Train
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "../../assets/LLMs/flan-t5-large", num_labels=2, id2label=id2label, label2id=label2id
)
ori_p = print_number_of_trainable_model_parameters(model)
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=None,  # peft will figure out the correct target_modules based on the model_type
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
peft_p = print_number_of_trainable_model_parameters(model)
print(f"# Trainable Parameter \nBefore: {ori_p} \nAfter: {peft_p} \nPercentage: {round(peft_p / ori_p * 100, 2)}")

training_args = TrainingArguments(
    output_dir="../../assets/Tuning/imdb_tuning_model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()