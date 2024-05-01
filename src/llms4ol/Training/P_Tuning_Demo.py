from Useful_Function_For_Training import *

from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
import datasets
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)

max_length = 512
device_map = "auto"
batch_size = 8
micro_batch_size = 2
gradient_accumulation_steps = batch_size // micro_batch_size

# load model
model = T5ForConditionalGeneration.from_pretrained("../../assets/LLMs/flan-t5-large", device_map=device_map)

tokenizer = T5Tokenizer.from_pretrained("../../assets/LLMs/flan-t5-large", use_fast=False, legacy=True,
                                        trust_remote_code=True)
ori_p = print_number_of_trainable_model_parameters(model)

# LoRA config
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=None,  # peft will figure out the correct target_modules based on the model_type
    bias="none",
    task_type="MASKED_LM",
)
model = get_peft_model(model, peft_config)

# compare trainable parameters #
peft_p = print_number_of_trainable_model_parameters(model)
print(f"# Trainable Parameter \nBefore: {ori_p} \nAfter: {peft_p} \nPercentage: {round(peft_p / ori_p * 100, 2)}")

file_path = '../../assets/Datasets/SubTaskB.1-GeoNames/geoname_train_pairs.json' # 请确保这个路径指向JSON文件
json_content = read_json_file(file_path)
print(json_content)

# #TODO
# train_data = []
# val_data = []
#
# args = TrainingArguments(
#     output_dir="../assets/Tuning/",
#     num_train_epochs=50,
#     max_steps=30,
#     fp16=True,
#     optim="paged_adamw_8bit",
#     learning_rate=2e-4,
#     lr_scheduler_type="constant",
#     per_device_train_batch_size=micro_batch_size,
#     gradient_accumulation_steps=gradient_accumulation_steps,
#     gradient_checkpointing=True,
#     group_by_length=False,
#     logging_steps=10,
#     save_strategy="epoch",
#     save_total_limit=3,
#     disable_tqdm=False,
# )
# trainer = Trainer(
#     model=model,
#     train_dataset=train_data,
#     eval_dataset=val_data,
#     args=args,
#     data_collator=DataCollatorForTokenClassification(
#         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
# )
#
# with torch.no_grad():
#     model.config.use_cache = False  # silence the warnings. re-enable for inference!
#     trainer.train()
#     model.save_pretrained("../assets/Tuning/flan-t5-large_tuned111")
