from transformers import pipeline
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from llms4ol.DataProcess.GeoNames.simple_ds_preprocess import *
from llms4ol.Evaluate.evaluate_metrics import EvaluationMetrics
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import torch

model_path = "../../../assets/LLMs/flan-t5-base"
peft_path = "../../../assets/Tuning/geo_tuning_model/checkpoint-19040"
train_dataset_path = "../../../assets/Datasets/SubTaskB.1-GeoNames/geoname_train_pairs.json"
eval_dataset_path = "../../../assets/Datasets/SubTaskB.1-GeoNames/test_dataset/hierarchy_test.json"
label, text, context = train_data_handler(train_dataset_path)
eval_label, eval_text = eval_data_handler(eval_dataset_path)
dataset = DatasetDict({'train': Dataset.from_dict({'label': label, 'text': text, 'context': context}),
                       'eval': Dataset.from_dict({'label': eval_label, 'text': eval_text})})
text = dataset["eval"][400]["text"]
print(dataset["eval"][400]["text"],dataset["eval"][400]["label"])
# loading model
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    use_cache=False,
    device_map="cuda"
)

# loading peft weight
model = PeftModel.from_pretrained(
    model,
    peft_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    print(model.config.id2label)
    print(model.config.id2label[predicted_class_id])