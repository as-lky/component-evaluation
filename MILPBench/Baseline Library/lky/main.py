import torch
import argparse
from lib.mod import Init2Preprocess
from lib.preprocess import Preprocess
from lib.graphencode import Graphencode
from lib.predict import Predict
from lib.modify import Modify
from lib.search import Search

parser = argparse.ArgumentParser(description=" receive select instruction from higher level")
parser.add_argument("--device", required=True, choices=["cpu", "cuda"], help="cpu or cuda")
parser.add_argument("--taskname", required=True, choices=["IP, IS, WA, CA"], help="taskname")
parser.add_argument("--instance_path", required=True, help="the task instance input path")
parser.add_argument("--train_instances_folder", help="the train instances input folder")

# log?
parser.add_argument("--graphencode", required=True, choices=["bi, tri"], help="graph encode component")
parser.add_argument("--predict", required=True, choices=["gcn"], help="predict component")
# search for model firstly

parser.add_argument("--modify", required=True, choices=["np"], help="modify component")
parser.add_argument("--search", required=True, choices=["gurobi", "scip"], help="search component")

def get_sequence_name():
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    sequence_name = get_sequence_name()
    # preprocess_component = Preprocess(args.device, args.taskname, args.instance_path, sequence_name)
    # graphencode_component = Graphencode(args.graphencode, device=args.device, taskname=args.taskname, instance=args.instance_path, sequence_name=sequence_name)
    # predict_component = Predict()
    # modify_component = Modify()
    # search_component = Search()
    
    # init2preprocess = Init2Preprocess(args.device, args.taskname, args.instance_input_folder)
    # preprocess2predict = preprocess_component.work(init2preprocess)
#    predict2modify = preprocess2predict.work()
#    modify2search = 