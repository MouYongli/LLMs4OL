from transformers import T5Tokenizer, T5ForConditionalGeneration,DebertaV2ForMaskedLM
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

    #MaskedLM -> Task A
    #SequenceClassification -> Task B

    
    #/flan-t5-xl
    # model = AutoModel.from_pretrained(model_id2).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, legacy=True,
    #                                           trust_remote_code=True)
    # 
    # input_text = prompt
    # input = tokenizer(input_text, return_tensors="pt").to("cuda")
    # with torch.no_grad():
    #     logits = model(**input).logits
    # 
    # mask_token_index = (input.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # 
    # predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    # print(tokenizer.decode(predicted_token_id))
    #outputs = model(**input, do_sample=True, max_new_tokens=600, temperature=0.8, top_p=0.8,
    #                       repetition_penalty=1.1)
    #print(tokenizer.decode(outputs[0]).replace("<pad>", "").replace("</s>", "").strip())

    #deberta-v3-large
    model = DebertaV2ForMaskedLM.from_pretrained(model_id2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, legacy=True,
                                             trust_remote_code=True)
    input_text = prompt
    input = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**input).logits

    mask_token_index = (input.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print(tokenizer.decode(predicted_token_id))
    

if __name__ == "__main__":
    generation("Perform a sentence completion on the following sentence:\nSentence: Pic de Font Blanca geographically is a [MASK]")