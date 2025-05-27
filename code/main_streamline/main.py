import os
import argparse
import time
import json
import re
from preprocess import Preprocess
from graphencode import Graphencode
from predict import Predict
from modify import Modify
from search import Search

parser = argparse.ArgumentParser(description="receive select instruction from higher level")
parser.add_argument("--device", required=True, choices=["cpu", "cuda", "cuda:2", "cuda:1", "cuda:3"], help="cpu or cuda")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "MKS", "SC"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance input path")
parser.add_argument("--train_data_dir", type=str, help="the train instances input folder")
parser.add_argument("--whole_time_limit", type=int, help="time limit for whole process")

parser.add_argument("--graphencode", required=True, choices=["bi", "tri", "bir", "trir", "default", "test"], help="graph encode component")
parser.add_argument("--predict", required=True, choices=["gat", "gcn", "gurobi", "scip", "cplex"], help="predict component")
parser.add_argument("--predict_time_limit", type=int, help="time limit for predicting")

parser.add_argument("--modify", required=True, choices=["sr", "nr", "np", "default"], help="modify component")
parser.add_argument("--modify_time_limit", type=int, help="time limit for modifying")
parser.add_argument("--search", required=True, choices=["gurobi", "scip", "LIH", "MIH", "LNS", "NALNS", "ACP"], help="search component")
parser.add_argument("--search_time_limit", type=int, help="time limit for searching")

parser.add_argument("--search_ACP_LNS_block", type=int, help="ACP / LNS block parameter")
parser.add_argument("--search_ACP_LNS_max_turn_ratio", type=float, help="ACP / LNS max_turn_ratio parameter")
parser.add_argument("--search_ACP_max_turn", type=float, help="ACP max_turn parameter")

parser.add_argument("--search_LIH_MIH_NALNS_choose", type=float, help="LIH / MIH / NALNS choose parameter")
parser.add_argument("--search_LIH_MIH_set_pa", type=float, help="LIH / MIH set_pa parameter")

parser.add_argument("--benchmark_path", type=str, help="benchmark path if use gurobi as benchmark")

# get algorithms' sequence name based on the selected components and hyper-parameters
def get_sequence_name(args):
    A = [args.graphencode, args.predict, args.modify, args.search]
    if args.search_ACP_LNS_block is not None:
        A.append(str(args.search_ACP_LNS_block))
    if args.search_ACP_LNS_max_turn_ratio is not None:
        A.append(str(args.search_ACP_LNS_max_turn_ratio))
    if args.search_LIH_MIH_NALNS_choose is not None:
        A.append(str(args.search_LIH_MIH_NALNS_choose))
    if args.search_LIH_MIH_set_pa is not None:
        A.append(str(args.search_LIH_MIH_set_pa))
    return A

if __name__ == "__main__":

    result_list_obj_time = []

    start_time = time.time()
    args = parser.parse_args()
    sequence_name = get_sequence_name(args)
    instance_name = os.path.basename(args.instance_path)
    whole_time_limit = args.whole_time_limit
    
    # create component instance in all layers
    preprocess_component = Preprocess(args.device, args.taskname, args.instance_path, sequence_name)
    graphencode_component = Graphencode(args.graphencode, args.device, args.taskname, args.instance_path, sequence_name)
    predict_component = Predict(args.predict, args.device, args.taskname, args.instance_path, sequence_name, time_limit=args.predict_time_limit, train_data_dir=args.train_data_dir)
    modify_component = Modify(args.modify, args.device, args.taskname, args.instance_path, sequence_name, time_limit=args.modify_time_limit)
    search_component = Search(args.search, args.device, args.taskname, args.instance_path, sequence_name, time_limit=args.search_time_limit, benchmark_path=args.benchmark_path, block=args.search_ACP_LNS_block, max_turn_ratio=args.search_ACP_LNS_max_turn_ratio, choose=args.search_LIH_MIH_NALNS_choose, set_pa=args.search_LIH_MIH_set_pa)
    
    # preprocess works and creates the directory
    preprocess_component.work()
    if whole_time_limit is not None:
       predict_component.time_limit = whole_time_limit - (time.time() - start_time)      
    now = 1
    
    # predict works and get the initial solution (maybe infeasible)
    now = predict_component.work(now)
    if whole_time_limit is not None:
        modify_component.time_limit = whole_time_limit - (time.time() - start_time)      
    # repair works and get the initial feasible solution
    now = modify_component.work(now)
    
    result_list_obj_time.append((time.time() - start_time, now.objval))    
    if whole_time_limit is not None:
       search_component.time_limit = whole_time_limit - (time.time() - start_time)      
    # search works and search for better solutions
    now = search_component.work(now, result_list_obj_time)

    # now : gap, obj, type (it's only meaningful when running with gurobi)
    gap = now[0]
    obj = now[1]
    type = -22222
    if len(now) == 3:
        type = now[2]
    
    # get the result and output it into file
    sn = ""
    for _ in sequence_name:
        sn += _ + "_"
    tmp = re.match(r"(.*)\.lp", instance_name)
    tmp = tmp.group(1)
    tmp_ = re.match(r"(.*)_[0-9]+", tmp).group(1)
    des = f'./logs/work/{args.taskname}/{sn}/{tmp_}/{tmp}_result.txt'

    result = {}
    result['gap'] = round(gap * 100, 4)
    result['obj'] = round(obj, 4)
    result['type'] = "" if type == -22222 else type
    result['result_list'] = [ (round(float(result_list_obj_time[i][0]), 8), round(float(result_list_obj_time[i][1]), 8)) for i in range(len(result_list_obj_time))]
    print(result)

    with open(des, 'w') as f:
        json.dump(result, f, indent=4)

    print("the result has been saved!")
    