import torch
import argparse
import time
from lib.mod import Init2Preprocess
from lib.preprocess import Preprocess
from lib.graphencode import Graphencode
from lib.predict import Predict
from lib.modify import Modify
from lib.search import Search

parser = argparse.ArgumentParser(description=" receive select instruction from higher level")
parser.add_argument("--device", required=True, choices=["cpu", "cuda"], help="cpu or cuda")
parser.add_argument("--taskname", required=True, choices=["IP", "IS", "WA", "CA"], help="taskname")
parser.add_argument("--instance_path", required=True, help="the task instance input path")
parser.add_argument("--train_instances_folder", help="the train instances input folder")

# log?
parser.add_argument("--graphencode", required=True, choices=["bi", "tri"], help="graph encode component")
parser.add_argument("--predict", required=True, choices=["gcn"], help="predict component")
# search for model firstly

parser.add_argument("--modify", required=True, choices=["np"], help="modify component")
parser.add_argument("--search", required=True, choices=["gurobi", "scip"], help="search component")

def get_sequence_name():
    return "test"

if __name__ == "__main__":

    start_time = time.time()
    args = parser.parse_args()
    sequence_name = get_sequence_name()
    preprocess_component = Preprocess(args.device, args.taskname, args.instance_path, sequence_name)
    graphencode_component = Graphencode(args.graphencode, args.device, args.taskname, args.instance_path, sequence_name)
    predict_component = Predict(args.predict, args.device, args.taskname, args.instance_path, sequence_name)
    modify_component = Modify(args.modify, args.device, args.taskname, args.instance_path, sequence_name)
    search_component = Search(args.search, args.device, args.taskname, args.instance_path, sequence_name)
    
    
    preprocess_component.work()
    now = graphencode_component.work()
    now = predict_component.work(now)
    now = modify_component.work(now)
    now = search_component.work(now)

    # gap
    gap = now
    end_time = time.time()
    
    des = f'./logs/{args.taskname}/{sequence_name}/work/result.txt'
    with open(des, 'w') as f:
        f.write(f"{gap * 100:0.4f} {end_time - start_time:0.4f}")
    print("the result has been saved!")
    
        
    
    
