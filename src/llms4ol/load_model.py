from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import AutoModel, AutoTokenizer

def generation(prompt):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(2))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    model_id = "meta-llama/Meta-Llama-3-8B"
    model_id1 = "google/flan-t5-xl"
    model_id2 = "microsoft/deberta-v3-large"
    
    #../assets/LLMs/flan-t5-large
    model = AutoModel.from_pretrained(model_id2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, legacy=True,
                                            trust_remote_code=True)

    input_text = prompt
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids, do_sample=True, max_new_tokens=600, temperature=0.8, top_p=0.8,
                             repetition_penalty=1.1)
    print(tokenizer.decode(outputs[0]).replace("<pad>", "").replace("</s>", "").strip())


if __name__ == "__main__":
    generation("translate English to German: How old are you?")