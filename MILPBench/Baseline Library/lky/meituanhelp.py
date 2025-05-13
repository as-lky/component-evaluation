import gurobipy as gp
import os
import pickle
import re
import shutil
#nn = "MT_hard_instance"
#instance_dir = "./Dataset/meituan_data/C_20000/C_20000/"
nn = "MT_2000_instance"
instance_dir = "./Dataset/meituan_data/C_2000/C_2000/"

to_dir = f"./Dataset/{nn}/{nn}"
num = 0
for i in os.listdir(instance_dir):
    if i.endswith(".lp"):
 #       print("!!!!!!!")
        name = os.path.splitext(os.path.basename(i))[0]
        instance = os.path.join(instance_dir, name + ".lp")
        pkl = os.path.join(instance_dir, name + ".pkl")
        if not os.path.exists(pkl):
            continue
        
        # with open(pkl, "rb") as f:
        #     d = pickle.load(f)
        #     for a, b in d.items():
        #         print(len(b))
        # break
        to_instance = os.path.join(to_dir, "LP", nn + "_" + str(num) + ".lp")
        to_pkl = os.path.join(to_dir, "Pickle", nn + "_" + str(num) + ".pickle")

        print(to_instance)
        print(to_pkl)
        
        shutil.copy(instance, to_instance)
#        shutil.copy(pkl, to_pkl)
        
        num += 1
        if num == 30:
            break
#NUM = 0
#q = 0
#print(len(os.listdir(instance_dir)))
#for i in os.listdir(instance_dir):
    # q += 1
    # if i.endswith(".lp"):
    #     model = gp.read(os.path.join(instance_dir, i))
    #     num = 0
    #     for j in model.getVars():
    #         num += 1
    #     NUM = max(NUM, num)
    # if q == 50:
    #     break
#print(NUM)

# C100 10000
# C200 20000
# C500 50000
# C1000 100000