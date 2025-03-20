import torch
import argparse
from mod import Init2Preprocess
from preprocess import Preprocess
from graphencode import Graphencode
from predict import Predict
from modify import Modify
from search import Search

parser = argparse.ArgumentParser(description=" receive select instruction from higher level")
parser.add_argument("device", choices=["cpu", "cuda"], help="cpu or cuda")
parser.add_argument("taskname", choice=["IP, IS, WA, CA"], help="taskname")
parser.add_argument("instance_input_folder", help="the task instance input path")
# log?
parser.add_argument("graphencode", choices=["bi, tri"], help="graph encode component")
parser.add_argument("predict", choices=["gcn"], help="predict component")
parser.add_argument("modify", choices=["default"], help="modify component")
parser.add_argument("search", choices=["default"], help="search component")

if __name__ == "__main__":
    args = parser.parse_args()
    preprocess_component = Preprocess()
    graphencode_component = Graphencode()
    predict_component = Predict()
    modify_component = Modify()
    search_component = Search()
    
    init2preprocess = Init2Preprocess(args.device, args.taskname, args.instance_input_folder)
    preprocess2predict = preprocess_component.work(init2preprocess)
#    predict2modify = preprocess2predict.work()
#    modify2search = 