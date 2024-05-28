import argparse
from llms4ol.Training.trainer import *
import multiprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--kb_name", required=True)     # geonames
    #parser.add_argument("--model", required=True)       # t5
    #parser.add_argument("--methode", required=True)       # lora
    args = parser.parse_args()

    print("args:", args)

    #taskB_trainer(args.model,args.kb_name,args.methode)
    methodes = ["lora","full"]
    kb_names = ["geonames"]
    models = ["roberta","llama3","t5"]
    
    for method in methodes:
        for model in models:
            for kb_name in kb_names:
                pass
                #taskB_trainer(model,kb_name,method)

    #taskB_trainer("llama3","geonames","full")
    #train model in multiprocesses
    for method in methodes:
        processes = []
        #Multiprocess
        for model in models:
            for kb_name in kb_names:
                process = multiprocessing.Process(target=taskB_trainer, args=(model,kb_name,method,))
                processes.append(process)
                process.start()
        for process in processes:
            process.join()
        print("All workers have finished")
