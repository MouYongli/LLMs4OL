from llms4ol.DataProcess.Data_Preprocess import *
import argparse


if __name__ == "__main__":
    #num define the number of subprocesses and number of parts of collected data

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)     # A | B
    parser.add_argument("--num", required=True)       # 1,2,3,4 | "GeoNames","Schema.org","UMLS","Gene Ontology"
    args = parser.parse_args()
    args.num = int(args.num)
    print("args:", args)
    if args.task not in {"A", "B"}:
        raise ValueError("Invalid value. Task must be 'A' or 'B'.")
    if args.num not in {1,2,3,4}:
        raise ValueError("Invalid value. Task_num must be 1 OR 2 OR 3 OR 4 ")

    print("args:", args)

    if args.task == "B":
            taskB_preprocess(args)


    