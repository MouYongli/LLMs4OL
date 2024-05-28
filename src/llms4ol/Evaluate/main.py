import argparse
from llms4ol.Evaluate.evaluater import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--kb_name", required=True)     # geonames
    #parser.add_argument("--model", required=True)       # t5
    #parser.add_argument("--methode", required=True)       # lora
    #parser.add_argument("--peft_path", required=False)       # path:""
    #parser.add_argument("--trained_model_path", required=False)       # path:""
    args = parser.parse_args()
    print("args:", args)
    #taskB_evaluater(args.model,args.kb_name,args.methode,args.peft_path,args.trained_model_path)


    methodes = ["lora","full"]
    kb_names = ["geonames"]
    models = ["roberta","llama3","t5"]
    for methode in methodes:
        for model in models:
            for kb_name in kb_names:
                taskB_evaluater(model,kb_name,methode)
                
