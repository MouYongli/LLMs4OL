import argparse
from llms4ol.Training.trainer import *
import multiprocessing
from llms4ol.path import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--kb_name", required=True)     # geonames
    #parser.add_argument("--model", required=True)       # t5
    #parser.add_argument("--methode", required=True)       # lora
    args = parser.parse_args()

    print("args:", args)

    #taskB_trainer(args.model,args.kb_name,args.methode)
    methods = ["continue"] # "lora","full","continue"
    kb_names = ["schema"]# "geonames","umls","wordnet","go" | "geonames","umls","schema","go"
    models = ["llama3"]# "roberta","llama3","t5"
    dataset_build_methods = [2] # 1,2,3,4


    #trained_model_path = output_dir="/home/yxpeng/DATA/Checkpoints/TaskA/WordNet/Finetune/llama3_Textclf_with_Context_full/checkpoint-*"
    #model_path = find_trained_model_path(trained_model_path)
    #taskA_trainer("llama3","wordnet","full",model_path)
    # multiprocessing.set_start_method('spawn')
    for model in models:
        for kb_name in kb_names:
            for method in methods:
                for dataset_build_method in dataset_build_methods:
                    taskB_trainer(model,kb_name,method,dataset_build_method)
                


    # # code for continuely train model:
    # for model in models:
    #     for kb_name in kb_names:
    #         for method in methods:
    #             for dataset_build_method in dataset_build_methods:
    #                 # m3 trained then train the model continuely with m2
    #                 if dataset_build_method == 2:
    #                     if model == "roberta":
    #                         if kb_name == "geonames":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/roberta_Textclf_with_Context_m3_full/checkpoint-*"
    #                         elif kb_name == "schema":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/roberta_Textclf_with_Context_m3_full/checkpoint-*"
    #                         elif kb_name == "umls":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/roberta_Textclf_with_Context_m3_full/checkpoint-*"
    #                         elif kb_name == "go":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/roberta_Textclf_with_Context_m3_full/checkpoint-*"
    #                         model_path = find_trained_model_path(path_pattern)
    #                     elif model == "llama3":
    #                         if kb_name == "geonames":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/llama3_Textclf_with_Context_m3_full/checkpoint-*"
    #                         elif kb_name == "schema":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/llama3_Textclf_with_Context_m3_full/checkpoint-*"
    #                         elif kb_name == "umls":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/llama3_Textclf_with_Context_m3_full/checkpoint-*"
    #                         elif kb_name == "go":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/llama3_Textclf_with_Context_m3_full/checkpoint-*"
    #                         model_path = find_trained_model_path(path_pattern)
    #                     elif model == "t5":
    #                         if kb_name == "geonames":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/flan-t5-xl_Textclf_with_Context_m3_full/checkpoint-*"
    #                         elif kb_name == "schema":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/flan-t5-xl_Textclf_with_Context_m3_full/checkpoint-*"
    #                         elif kb_name == "umls":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/flan-t5-xl_Textclf_with_Context_m3_full/checkpoint-*"
    #                         elif kb_name == "go":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/flan-t5-xl_Textclf_with_Context_m3_full/checkpoint-*"
    #                         model_path = find_trained_model_path(path_pattern)
    #                 # m2 trained then train the model continuely with m3
    #                 if dataset_build_method == 3:
    #                     if model == "roberta":
    #                         if kb_name == "geonames":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/roberta_Textclf_with_Context_m2_full/checkpoint-*"
    #                         elif kb_name == "schema":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/roberta_Textclf_with_Context_m2_full/checkpoint-*"
    #                         elif kb_name == "umls":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/roberta_Textclf_with_Context_m2_full/checkpoint-*"
    #                         elif kb_name == "go":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/roberta_Textclf_with_Context_m2_full/checkpoint-*"
    #                         model_path = find_trained_model_path(path_pattern)
    #                     elif model == "llama3":
    #                         if kb_name == "geonames":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/llama3_Textclf_with_Context_m2_full/checkpoint-*"
    #                         elif kb_name == "schema":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/llama3_Textclf_with_Context_m2_full/checkpoint-*"
    #                         elif kb_name == "umls":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/llama3_Textclf_with_Context_m2_full/checkpoint-*"
    #                         elif kb_name == "go":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/llama3_Textclf_with_Context_m2_full/checkpoint-*"
    #                         model_path = find_trained_model_path(path_pattern)
    #                     elif model == "t5":
    #                         if kb_name == "geonames":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GeoNames/Finetune/flan-t5-xl_Textclf_with_Context_m2_full/checkpoint-*"
    #                         elif kb_name == "schema":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/Schema/Finetune/flan-t5-xl_Textclf_with_Context_m2_full/checkpoint-*"
    #                         elif kb_name == "umls":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/UMLS/Finetune/flan-t5-xl_Textclf_with_Context_m2_full/checkpoint-*"
    #                         elif kb_name == "go":
    #                             path_pattern = f"/home/yxpeng/DATA/Checkpoints/TaskB/GO/Finetune/flan-t5-xl_Textclf_with_Context_m2_full/checkpoint-*"
    #                         model_path = find_trained_model_path(path_pattern)
    #                 taskB_trainer(model,kb_name,method,dataset_build_method,model_path)