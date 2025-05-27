import os
import argparse
import re
import subprocess
import json

parser = argparse.ArgumentParser(description="receive evaluate_delta instruction")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "SC", "MKS"], help="taskname")
args = parser.parse_args()

# get the hypervolume score of some algorithms' perform vectors on a constant instance refering to (1, 1, ..., 1)
def calc(lis):
    # lis is a list in which each element is a list, corresponding to a point vector meaning an algorithm's performance on an instance
    # each dim of the vector is a float number in [0, 1)
    if os.path.exists('../calc/hbda/build/nonincremental/tmp.txt'):
        os.remove('../calc/hbda/build/nonincremental/tmp.txt')
    SUM = 0
    # write the list input into a tmp file
    with open('../calc/hbda/build/nonincremental/tmp.txt', 'w') as f:
        for i in lis:
            for j in i:
                w = round(j, 8)
                f.write(f"{w} ")
            f.write("\n")
    if os.path.exists('../calc/hbda/build/nonincremental/result.txt'):
        os.remove('../calc/hbda/build/nonincremental/result.txt')

    # run the hypervolume calculation program
    subprocess.run(['../calc/hbda/build/nonincremental/nonincremental', '-O', '../calc/hbda/build/nonincremental/tmp.txt', '-S', '../calc/hbda/build/nonincremental/result.txt'])

    # read the result from the output file
    with open('../calc/hbda/build/nonincremental/result.txt', 'r') as f:
        lines = f.readlines()
        line_result = lines[1].strip()
        result = line_result.split()
        SUM = float(result[-1])
    return SUM

# return true if a dominates b
def domi(a, b):
    l = len(a)
    for i in range(l):
        if a[i] > b[i]:
            return False
    return True 

# get the instance list for evaluation through taskname
if args.taskname == "IS":
    INSLIST = ["IS_easy_instance_0", "IS_easy_instance_1", "IS_easy_instance_2"]
if args.taskname == "SC":
    INSLIST = ["SC_easy_instance_0", "SC_easy_instance_1", "SC_easy_instance_2"]
if args.taskname == "MVC":
    INSLIST = ["MVC_easy_instance_0", "MVC_easy_instance_1", "MVC_easy_instance_2"]
if args.taskname == "MKS":
    INSLIST = ["MKS_easy_instance_0", "MKS_easy_instance_1", "MKS_easy_instance_2"]
    
# get the algorithms' perform result from eval.py output path
instance_name = INSLIST[0]
tmp = re.match(r"(.*)_[0-9]+", instance_name)
tmp = tmp.group(1)
des = f'./logs/work/{args.taskname}/{tmp}_result.txt'
with open(des, 'r') as f:
    result_list = json.load(f)

# get each algorithm's Hypervolume delta score
score_dic = {}
SCORES = {}
for a, b in result_list.items():
    SCORES[a] = 0

for instance in INSLIST:
    # get Pareto Envelop
    tmp = []
    for i, value in result_list.items():
        if instance not in value:
            continue
        flag = 0
        for j, value2 in result_list.items():
            if i == j or instance not in value2:
                continue
            if domi(value2[instance][:-1], value[instance][:-1]):
                flag = 1
                break
        if flag == 0:
            tmp.append(i)
    
    # calculate the delta score for each algorithm
    if len(tmp) == 1:
        SCORES[tmp[0]] += calc([result_list[tmp[0]][instance][:-1]]) * 1e7
    else:
        er = []
        for i in tmp:
            er.append(result_list[i][instance][:-1])
        score = calc(er)
        for i in tmp:
            po = er.copy()
            po.remove(result_list[i][instance][:-1])
            SCORES[i] += (score - calc(po)) * 1e7
            
# write the scores to output file
des = f'../logs/work/{args.taskname}/{tmp}_result_delta.txt'
SCORES = dict(sorted(SCORES.items(), key=lambda x: x[1], reverse=True))
with open(des, 'w') as f:                
    json.dump(SCORES, f, indent=4)
