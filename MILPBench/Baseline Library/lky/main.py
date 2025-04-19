import torch
import os
import argparse
import time
import json
import re
from lib.mod import Init2Preprocess
from lib.preprocess import Preprocess
from lib.graphencode import Graphencode
from lib.predict import Predict
from lib.modify import Modify
from lib.search import Search

parser = argparse.ArgumentParser(description=" receive select instruction from higher level")
parser.add_argument("--device", required=True, choices=["cpu", "cuda"], help="cpu or cuda")
parser.add_argument("--taskname", required=True, choices=["IP", "IS", "WA", "CA"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
parser.add_argument("--train_data_dir", type=str, help="the train instances input folder")

# log?
parser.add_argument("--graphencode", required=True, choices=["bi", "tri", "bir", "trir", "default"], help="graph encode component")
parser.add_argument("--predict", required=True, choices=["l2bs", "gtran", "gat", "gcn", "gurobi", "scip", "cplex"], help="predict component")
parser.add_argument("--predict_time_limit", type=int, help="time limit for predicting")

# search for model firstly 

parser.add_argument("--modify", required=True, choices=["sr", "nr", "np", "default"], help="modify component")
parser.add_argument("--search", required=True, choices=["gurobi", "scip", "LIH", "MIH", "LNS", "NALNS", "ACP"], help="search component")
parser.add_argument("--search_time_limit", type=int, help="time limit for searching")

def get_sequence_name(args):
    return [args.graphencode, args.predict, args.modify, args.search]

if __name__ == "__main__":

    result_list_obj_time = []

    start_time = time.time()
    args = parser.parse_args()
    sequence_name = get_sequence_name(args)
    instance_name = os.path.basename(args.instance_path)
    preprocess_component = Preprocess(args.device, args.taskname, args.instance_path, sequence_name)
    graphencode_component = Graphencode(args.graphencode, args.device, args.taskname, args.instance_path, sequence_name)
    predict_component = Predict(args.predict, args.device, args.taskname, args.instance_path, sequence_name, time_limit=args.predict_time_limit, train_data_dir=args.train_data_dir)
    modify_component = Modify(args.modify, args.device, args.taskname, args.instance_path, sequence_name)
    search_component = Search(args.search, args.device, args.taskname, args.instance_path, sequence_name, time_limit=args.search_time_limit)
    
    preprocess_component.work()
    now = graphencode_component.work()
    now = predict_component.work(now)
    now = modify_component.work(now)
    
    result_list_obj_time.append((time.time() - start_time, now.objval))    

    now = search_component.work(now, result_list_obj_time)

    # now : gap, obj, type
    gap = now[0]
    obj = now[1]
    type = -22222
    if len(now) == 3:
        type = now[2]
    
    sn = ""
    for _ in sequence_name:
        sn += _ + "_"
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    des = f'./logs/work/{args.taskname}/{sn}/{tmp}_result.txt'

    result = {}
    result['gap'] = round(gap * 100, 4)
    result['obj'] = round(obj, 4)
    result['type'] = "" if type == -22222 else type
    result['result_list'] = [ (float(round(result_list_obj_time[i][0], 8)), float(round(result_list_obj_time[i][1], 8))) for i in range(len(result_list_obj_time))]
    print(result)

    with open(des, 'w') as f:
        json.dump(result, f, indent=4)

    print("the result has been saved!")
    
    # python main.py --device cuda --taskname IS --instance_path ./Dataset/IS_easy_instance/IS_easy_instance/LP/IS_easy_instance_0.lp --graphencode tri --predict gcn --train_data_dir ./Dataset/IS_easy_instance/IS_easy_instance/ --modify sr --search gurobi