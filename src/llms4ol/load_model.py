from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


def generation(prompt):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(2))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    #../assets/LLMs/flan-t5-large
    model = T5ForConditionalGeneration.from_pretrained("/home/yxpeng/DATA/flan-t5-xl").to(device)
    tokenizer = T5Tokenizer.from_pretrained("/home/yxpeng/DATA/flan-t5-xl", use_fast=False, legacy=True,
                                            trust_remote_code=True)

    input_text = prompt
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids, do_sample=True, max_new_tokens=600, temperature=0.8, top_p=0.8,
                             repetition_penalty=1.1)
    print(tokenizer.decode(outputs[0]).replace("<pad>", "").replace("</s>", "").strip())


if __name__ == "__main__":
    generation("translate English to German: How old are you?")