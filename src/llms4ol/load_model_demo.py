from transformers import T5Tokenizer, T5ForConditionalGeneration,DebertaV2ForMaskedLM
import torch,transformers
from transformers import AutoModel, AutoTokenizer,LlamaForCausalLM,AutoModelForSeq2SeqLM

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
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_id1).to(device)
    #tokenizer = AutoTokenizer.from_pretrained(model_id1, use_fast=False, legacy=True,
     #                                         trust_remote_code=True)
   # input_text = prompt
    #input = tokenizer(input_text, return_tensors="pt").to('cuda:0')
    #outputs = model.generate(**input)
   # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    #deberta-v3-large
    #model = AutoModel.from_pretrained(model_id2).to(device)
    #tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, legacy=True, trust_remote_code=True)
    #input_text = prompt
    #inputs = tokenizer(prompt, return_tensors="pt").input_ids
    #outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    #print(tokenizer.decode(outputs, skip_special_tokens=True))
    #mask_token_index = (input.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    #predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    #print(tokenizer.decode(predicted_token_id))


    #llama3-70B
    # !pip install -U "transformers==4.40.0" --upgrade
    model_id = "meta-llama/Meta-Llama-3-70B"
    name = "Allen, Mount"
    prompt_templete = f'''
     Here is a geographical name: {name}, translate it into english, give the geographical information in plain text without any markdown format. 
    No reference link in result. 
    Make sure all provided information can be used for discovering implicit relation of other geographical term, but don't mention the relation in result.
     '''
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, legacy=True)
    model = LlamaForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16, device="cuda")
    inputs = tokenizer(prompt_templete, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1280, repetition_penalty=1.1)
    outputs = tokenizer.decode(outputs.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(outputs)



if __name__ == "__main__":
    generation("Pic de Font Blanca geographically is a:")