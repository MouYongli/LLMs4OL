from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
import datasets
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    GenerationConfig
)


model_id = "../assets/LLMs/flan-t5-large"
peft_path = "../assets/Tuning/flan-t5-large_tuned111"

# loading model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("../../assets/LLMs/flan-t5-large", device_map="auto")
tokenizer = T5Tokenizer.from_pretrained("../../assets/LLMs/flan-t5-large", use_fast=False, legacy=True,
                                        trust_remote_code=True)

# loading peft weight
model = PeftModel.from_pretrained(
    model,
    peft_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# generation config
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4, # beam search, choose candidate_sequences(beams)
)

# generating reply
with torch.no_grad():
    prompt = "Say Something"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generation_output = model.generate(
        input_ids=inputs.input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=300,
    )
    print('\nAnswer: ', tokenizer.decode(generation_output.sequences[0]))