from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model,PeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification,T5Tokenizer,T5ForConditionalGeneration
from llms4ol.Evaluate.evaluate_metrics import EvaluationMetrics
from llms4ol.DataProcess.Dataset_Builder import GeoNames_DataBuilder,GO_DataBuilder,Schema_DataBuilder,UMLS_DataBuilder,WordNet_DataBuilder
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import torch,random
from tqdm import tqdm
from llms4ol.path import *

def taskA_evaluater(model_name,kb_name):
    root_path = find_root_path()
    def random_sample_corresponding(list1, list2, list3, sample_size):
        if len(list3) == 0:
            combined_list = list(zip(list1, list2))
            # Step 2: Perform random sampling on the combined list
            sampled_combined_list = random.sample(combined_list, sample_size)
            
            # Step 3: Unzip the sampled list of tuples back into lists
            sampled_list1, sampled_list2 = zip(*sampled_combined_list)
            
            return list(sampled_list1), list(sampled_list2)
        else:
            combined_list = list(zip(list1, list2, list3))

            # Step 2: Perform random sampling on the combined list
            sampled_combined_list = random.sample(combined_list, sample_size)
            
            # Step 3: Unzip the sampled list of tuples back into lists
            sampled_list1, sampled_list2, sampled_list3 = zip(*sampled_combined_list)
            
            return list(sampled_list1), list(sampled_list2), list(sampled_list3)

    if model_name == "roberta":
        if kb_name == "geonames":
            id2label,label2id,text,label = GeoNames_DataBuilder.Geo_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Finetune/roberta_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "wordnet":
            id2label,label2id,text,label,sentences = WordNet_DataBuilder.WN_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label,sentences = random_sample_corresponding(text, label, sentences, int(0.2*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/roberta_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "umls":
            id2label,label2id,text,label = UMLS_DataBuilder.UMLS_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Finetune/roberta_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "go":
            id2label,label2id,text,label = GO_DataBuilder.GO_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GO/Finetune/roberta_Textclf_with_Context_full/checkpoint-*"
        model_path = find_trained_model_path(path_pattern)
    elif model_name == "llama3":
        if kb_name == "geonames":
            id2label,label2id,text,label = GeoNames_DataBuilder.Geo_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "wordnet":
            id2label,label2id,text,label,sentences = WordNet_DataBuilder.WN_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label,sentences = random_sample_corresponding(text, label, sentences, int(0.2*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "umls":
            id2label,label2id,text,label = UMLS_DataBuilder.UMLS_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "go":
            id2label,label2id,text,label = GO_DataBuilder.GO_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GO/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
        model_path = find_trained_model_path(path_pattern)
    elif model_name == "t5":
        if kb_name == "geonames":
            id2label,label2id,text,label = GeoNames_DataBuilder.Geo_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GeoNames/Finetune/flan-t5-xl_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "wordnet":
            id2label,label2id,text,label,sentences = WordNet_DataBuilder.WN_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label,sentences = random_sample_corresponding(text, label, sentences, int(0.2*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/flan-t5-xl_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "umls":
            id2label,label2id,text,label = UMLS_DataBuilder.UMLS_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/UMLS/Finetune/flan-t5-xl_Textclf_with_Context_full/checkpoint-*"
        elif kb_name == "go":
            id2label,label2id,text,label = GO_DataBuilder.GO_TaskA_TextClf_dataset_builder()
            # random sample on training dataset as test dataset
            text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
            path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/GO/Finetune/flan-t5-xl_Textclf_with_Context_full/checkpoint-*"
        model_path = find_trained_model_path(path_pattern)


    test_text = text
    test_label = label
    dataset = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text})})

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        label2id=label2id,
        id2label=id2label,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    actual_label = []
    predicted_label = []
    test_dataset = dataset["test"]
    for item in tqdm(test_dataset, desc = "Infernce Begin ", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        actual_label.append(item['label'])
        temp = item['text']
        if kb_name == "geonames":
            single_text = f"What is the geographical type of '{temp}'?"
        elif kb_name == "wordnet":
            single_text = f"{temp}"
        elif kb_name == "umls":
            single_text = f"{temp}"
        elif kb_name == "go":
            single_text = f"{temp}"
        inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item() #top 1
            predicted_label.append(predicted_class_id)
    results = EvaluationMetrics.evaluate_accuracy(actual=actual_label, predicted=predicted_label)
    file_path = root_path + '/src/llms4ol/Evaluate/TaskA_after_trained_evaluation_results.txt'
    content = f"{model_name} after training on KB {kb_name} for task A has:"
    content += "\nAccuracy: \n"
    content += str(results['accuracy'])
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(content + '\n\n')



def taskA_vanilla_evaluater(model_name,kb_name):
    root_path = find_root_path()
    def random_sample_corresponding(list1, list2, list3, sample_size):
        if len(list3) == 0:
            combined_list = list(zip(list1, list2))
            # Step 2: Perform random sampling on the combined list
            sampled_combined_list = random.sample(combined_list, sample_size)
            
            # Step 3: Unzip the sampled list of tuples back into lists
            sampled_list1, sampled_list2 = zip(*sampled_combined_list)
            
            return list(sampled_list1), list(sampled_list2)
        else:
            combined_list = list(zip(list1, list2, list3))

            # Step 2: Perform random sampling on the combined list
            sampled_combined_list = random.sample(combined_list, sample_size)
            
            # Step 3: Unzip the sampled list of tuples back into lists
            sampled_list1, sampled_list2, sampled_list3 = zip(*sampled_combined_list)
            
            return list(sampled_list1), list(sampled_list2), list(sampled_list3)
    
    if model_name == "roberta":
        model_path = "FacebookAI/roberta-large"
    elif model_name == "llama3":
        model_path = "meta-llama/Meta-Llama-3-8B"
    elif model_name == "t5":
        model_path = "google/flan-t5-xl"

    if kb_name == "wordnet":
        id2label,label2id,text,label,sentences = WordNet_DataBuilder.WN_TaskA_TextClf_dataset_builder()
        # random sample on training dataset as test dataset
        text,label,sentences = random_sample_corresponding(text, label, sentences, int(0.2*(len(text))))# int(0.2*(len(text)))
        test_text1 = []
        test_text2 = []
        test_text3 = []
        test_text4 = []
        test_text5 = []
        test_text6 = []
        test_text7 = text
        test_label = label
        for i, item in enumerate(text):
            element = "Please determine the lexical nature of word "
            element += f"'{item}'"
            test_text1.append(element)
            element += ", please answer with only noun, verb, adj, adv"
            test_text2.append(element)
            element = f"Perform a sentence completion on the following sentence: '{item}' part of speech is a ___.\nThe answer is"
            test_text3.append(element)
            element = f"Perform a sentence completion on the following sentence: {sentences[i]}. '{item}' part of speech is a ___.\nThe answer is"
            test_text4.append(element)
            element = f"'{item}' part of speech is a ?"
            test_text5.append(element)
            element = f"{sentences[i]}. '{item}' part of speech is a ?"
            test_text6.append(element)

        #Examples:
        #   test_text1 = "Please determine the lexical nature of word 'cover' "
        #   test_text2 = "Please determine the lexical nature of word 'cover', please answer with only noun, verb, adj, adv. "
        #   test_text3 = "Perform a sentence completion on the following sentence: 'cover' part of speech is a ___.\nThe answer is"
        #   test_text4 = "Perform a sentence completion on the following sentence: cover her face with a handkerchief. 'cover' part of speech is a ___.\nThe answer is"
        #   test_text5 = "'cover' part of speech is a ?"
        #   test_text6 = "cover her face with a handkerchief. 'cover' part of speech is a ?"
        #   test_text7 = "cover"
        # dataset3 will be applied for evaluate vanilla and finetuned model
        dataset1 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text1})})
        dataset2 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text2})})
        dataset3 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text3})})
        dataset4 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text4})})
        dataset5 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text5})})
        dataset6 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text6})})
        dataset7 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text7})})

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            label2id=label2id,
            id2label=id2label,
            device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        num = 0
        for dataset in [dataset1,dataset2,dataset3,dataset4,dataset5,dataset6,dataset7]:
            num += 1
            actual_label = []
            predicted_label = []
            test_dataset = dataset["test"]
            for item in tqdm(test_dataset, desc = "Infernce Begin ", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
                actual_label.append(item['label'])
                single_text = item['text']
                inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = model(**inputs).logits
                    predicted_class_id = logits.argmax().item() #top 1
                    predicted_label.append(predicted_class_id)
            results = EvaluationMetrics.evaluate_accuracy(actual=actual_label, predicted=predicted_label)
            prompt_temp = dataset["test"][0]['text']
            file_path = root_path + '/src/llms4ol/Evaluate/TaskA_evaluation_results.txt'
            content = f"Prompt Template is: {prompt_temp} \n"
            content += f"{model_name} on KB {kb_name} for task A on dataset{num} has:"
            content += "\nAccuracy: \n"
            content += str(results['accuracy'])
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(content + '\n\n')
    elif kb_name == "umls":
        id2label,label2id,text,label = UMLS_DataBuilder.UMLS_TaskA_TextClf_dataset_builder()
        # random sample on training dataset as test dataset
        text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
        test_text1 = text
        test_text2 = []
        test_text3 = []
        test_text4 = []
        test_text5 = []
        test_label = label
        for i, item in enumerate(text):
            element = f"'{item}' in medicine is a ?"
            test_text2.append(element)
            element = f"'{item}' in biomedicine is a ?"
            test_text3.append(element)
        dataset1 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text1})})
        dataset2 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text2})})
        dataset3 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text3})})

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            label2id=label2id,
            id2label=id2label,
            device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        num = 0
        for dataset in [dataset1,dataset2,dataset3]:
            num += 1
            actual_label = []
            predicted_label = []
            for item in dataset["test"]:
                actual_label.append(item['label'])
                single_text = item['text']
                inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = model(**inputs).logits
                    predicted_class_id = logits.argmax().item() #top 1
                    predicted_label.append(predicted_class_id)
            results = EvaluationMetrics.evaluate_accuracy(actual=actual_label, predicted=predicted_label)

            prompt_temp = dataset["test"][0]['text']
            file_path = root_path + '/src/llms4ol/Evaluate/TaskA_evaluation_results.txt'
            content = f"Prompt Template is: {prompt_temp}\n"
            content += f"{model_name} on KB {kb_name} for task A on dataset{num} has:"
            content += "\nAccuracy: \n"
            content += str(results['accuracy'])
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(content + '\n\n')
    elif kb_name == "go":
        id2label,label2id,text,label = GO_DataBuilder.GO_TaskA_TextClf_dataset_builder()
        # random sample on training dataset as test dataset
        text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
        test_text1 = text
        test_text2 = []
        test_text3 = []
        test_text4 = []
        test_text5 = []
        test_label = label
        for i, item in enumerate(text):
            element = f"'{item}' in medicine is a ?"
            test_text2.append(element)
            element = f"'{item}' in biomedicine is a ?"
            test_text3.append(element)
            element = f"What is the biological type of '{item}' in Gene Ontology ?"
            test_text4.append(element)
            element = f"The biological type of '{item}' is a ?"
            test_text5.append(element)
        dataset1 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text1})})
        dataset2 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text2})})
        dataset3 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text3})})
        dataset4 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text4})})
        dataset5 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text5})})

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            label2id=label2id,
            id2label=id2label,
            device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        num = 0
        for dataset in [dataset1,dataset2,dataset3,dataset4,dataset5]:
            num += 1
            actual_label = []
            predicted_label = []
            for item in dataset["test"]:
                actual_label.append(item['label'])
                single_text = item['text']
                inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = model(**inputs).logits
                    predicted_class_id = logits.argmax().item() #top 1
                    predicted_label.append(predicted_class_id)
            results = EvaluationMetrics.evaluate_accuracy(actual=actual_label, predicted=predicted_label)

            prompt_temp = dataset["test"][0]['text']
            file_path = root_path + '/src/llms4ol/Evaluate/TaskA_evaluation_results.txt'
            content = f"Prompt Template is: {prompt_temp}\n"
            content += f"{model_name} on KB {kb_name} for task A on dataset{num} has:"
            content += "\nAccuracy: \n"
            content += str(results['accuracy'])
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(content + '\n\n')
    elif kb_name == "geonames":
        id2label,label2id,text,label = GeoNames_DataBuilder.Geo_TaskA_TextClf_dataset_builder()
        # random sample on training dataset as test dataset
        text,label = random_sample_corresponding(text, label, [], int(0.1*(len(text))))# int(0.2*(len(text)))
        test_text1 = text
        test_text2 = []
        test_text3 = []
        test_text4 = []
        test_text5 = []
        test_label = label
        for i, item in enumerate(text):
            element = f"'{item}' geographically is a ?"
            test_text2.append(element)
            element = f"What is the geographical type of '{item}' ? Please answer with professional terminology."
            test_text3.append(element)
            element = f"What is the geographical type of '{item}' ?"
            test_text4.append(element)
        dataset1 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text1})})
        dataset2 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text2})})
        dataset3 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text3})})
        dataset4 = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text4})})

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            label2id=label2id,
            id2label=id2label,
            device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        num = 0
        for dataset in [dataset1,dataset2,dataset3,dataset4]:
            num += 1
            actual_label = []
            predicted_label = []
            for item in dataset["test"]:
                actual_label.append(item['label'])
                single_text = item['text']
                inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = model(**inputs).logits
                    predicted_class_id = logits.argmax().item() #top 1
                    predicted_label.append(predicted_class_id)
            results = EvaluationMetrics.evaluate_accuracy(actual=actual_label, predicted=predicted_label)

            prompt_temp = dataset["test"][0]['text']
            file_path = root_path + '/src/llms4ol/Evaluate/TaskA_evaluation_results.txt'
            content = f"Prompt Template is: {prompt_temp}\n"
            content += f"{model_name} on KB {kb_name} for task A on dataset{num} has:"
            content += "\nAccuracy: \n"
            content += str(results['accuracy'])
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(content + '\n\n')

    # loading peft weight
    if 0 :
        root_path = find_root_path()
        path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/llama3_Textclf_with_Context_lora/checkpoint-*"
        peft_path = find_trained_model_path(path_pattern)
        print(model_path, peft_path)
        model = PeftModelForSequenceClassification.from_pretrained(
            model,
            peft_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        num = 0
        for dataset in [dataset1,dataset2,dataset3,dataset4,dataset5]:
            num += 1
            actual_label = []
            predicted_label = []
            for item in dataset["test"]:
                actual_label.append(item['label'])
                single_text = item['text']
                inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = model(**inputs).logits
                    predicted_class_id = logits.argmax().item() #top 1
                    predicted_label.append(predicted_class_id)
            print(actual_label[:10])
            print(predicted_label[:10])
            results = EvaluationMetrics.evaluate_accuracy(actual=actual_label, predicted=predicted_label)

            file_path = root_path + '/src/llms4ol/Evaluate/TaskA_evaluation_results.txt'
            content = f"{model_name} with finetune methode lora on KB {kb_name} on dataset{num} for task A has: \n"
            content += "\nAccuracy: \n"
            content += str(results['accuracy'])
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(content + '\n')


def taskB_evaluater(method_num,model_name,kb_name,finetune_methode,peft_path=None,trained_model_path = None):

    if model_name == "roberta":
        model_path = "FacebookAI/roberta-large"
    elif model_name == "llama3":
        model_path = "meta-llama/Meta-Llama-3-8B"
    elif model_name == "t5":
        model_path = "google/flan-t5-xl"

    if finetune_methode == "lora" and peft_path:
        peft_path = peft_path
    elif finetune_methode == "full" and trained_model_path:
        model_path = trained_model_path
    elif finetune_methode == "lora" and peft_path == None:
        if model_name == "roberta":
            path_pattern = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/roberta_Context_lora/checkpoint-*"
            peft_path = find_trained_model_path(path_pattern)
        elif model_name == "llama3":
            path_pattern = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/llama3_Context_lora/checkpoint-*"
            peft_path = find_trained_model_path(path_pattern)
        elif model_name == "t5":
            path_pattern = "/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/flan-t5-xl_Context_lora/checkpoint-*"
            peft_path = find_trained_model_path(path_pattern)
    elif finetune_methode == "full" and trained_model_path == None:
        if model_name == "roberta":
            if kb_name == "geonames":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/roberta_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            elif kb_name == "schema":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/roberta_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            elif kb_name == "umls":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/roberta_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            elif kb_name == "go":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/roberta_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            model_path = find_trained_model_path(path_pattern)
        elif model_name == "llama3":
            if kb_name == "geonames":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/llama3_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            elif kb_name == "schema":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/llama3_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            elif kb_name == "umls":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/llama3_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            elif kb_name == "go":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/llama3_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            model_path = find_trained_model_path(path_pattern)
        elif model_name == "t5":
            if kb_name == "geonames":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/flan-t5-xl_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            elif kb_name == "schema":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/flan-t5-xl_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            elif kb_name == "umls":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/flan-t5-xl_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            elif kb_name == "go":
                path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/flan-t5-xl_Textclf_with_Context_m{method_num}_full/checkpoint-*"
            model_path = find_trained_model_path(path_pattern)


    root_path = find_root_path()
    if kb_name == "geonames":
        test_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.1-GeoNames/test_dataset/hierarchy_test.json"
        test_label, test_text = GeoNames_DataBuilder.Geo_TaskB_TextClf_test_dataset_builder(test_dataset_path)
    elif kb_name == "schema":
        test_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.2-Schema.org/test_dataset/hierarchy_test.json"
        test_label, test_text = Schema_DataBuilder.Schema_TaskB_TextClf_test_dataset_builder(test_dataset_path)
    elif kb_name == "umls":
        test_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.3-UMLS/test_dataset/hierarchy_test.json"
        test_label, test_text = UMLS_DataBuilder.UMLS_TaskB_TextClf_test_dataset_builder(test_dataset_path)
    elif kb_name == "go":
        test_dataset_path = root_path + "/src/assets/Datasets/SubTaskB.4-GO/go_train_pairs.json"
        test_label, test_text, test_context = GO_DataBuilder.GO_TaskB_TextClf_evl_dataset_builder(test_dataset_path)
        
    dataset = DatasetDict({'test': Dataset.from_dict({'label': test_label, 'text': test_text})})

    id2label = {0: "incorrect", 1: "correct"}
    label2id = {"incorrect": 0, "correct": 1}

    if model_path != None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            label2id=label2id,
            id2label=id2label,
            device_map="cuda"
        )
        print(model_path, peft_path)
        if finetune_methode =="lora":
            # loading peft weight
            model = PeftModelForSequenceClassification.from_pretrained(
                model,
                peft_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        actual_label = []
        predicted_label = []

        for item in dataset["test"]:
            actual_label.append(item['label'])
            single_text = item['text']
            inputs = tokenizer(single_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**inputs).logits
                predicted_class_id = logits.argmax().item() #top 1
                predicted_label.append(predicted_class_id)

        results = EvaluationMetrics.evaluate(actual=actual_label, predicted=predicted_label)


        file_path = root_path + '/src/llms4ol/Evaluate/evaluation_results.txt'
        if finetune_methode == "vanilla":
            content = f"{model_name} on KB {kb_name} for task B has: \n"
        else:
            content = f"{model_name} with finetune methode {finetune_methode} on KB {kb_name} for task B with data-construction method {method_num}  has: \n"
        content += "F1-score: \n"
        content += str(results['clf-report-dict']['macro avg']['f1-score'])
        content += "\nAccuracy: \n"
        content += str(results['accuracy'])
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(content + '\n')
        print(f"{model_name} with finetune methode {finetune_methode} on KB {kb_name} for task B with data-construction method {method_num}  has :\n")
        print("F1-score:", results['clf-report-dict']['macro avg']['f1-score'])
