import gurobipy as gp
import os
import pickle
import re
import shutil

#rr = "MT_hard_instance"
rr = "MT_medium_instance"
instance = f"./Dataset/{rr}/{rr}/LP/{rr}_0.lp"

model = gp.read(instance)
print(model.NumVars)
model.optimize()
#nn = "MT_hard_instance"
#instance_dir = "./Dataset/meituan_data/C_20000/C_20000/"
#nn = "MT_2000_instance"
#instance_dir = "./Dataset/meituan_data/C_2000/C_2000/"
# nn = "MT_2000_instance"
# mm = "MT_medium_instance"

# from_dir = f"./Dataset/{nn}/{nn}/" 
# to_dir = f"./Dataset/{mm}/{mm}/"


# for i in os.listdir(from_dir + "LP/"):
# #    if i.endswith(".lp"):
#     tmp = os.path.basename(i)
#     tmp = re.match(r".*_([0-9]+)", tmp)
#     tmp = tmp.group(1)

#     instance = os.path.join(to_dir + "LP/", mm + "_" + tmp + ".lp")
#     shutil.copy(from_dir + "LP/" + i, instance)

#     j = from_dir + "Pickle/" + nn + "_" + tmp + ".pickle"
#     if not os.path.exists(j):
#         print(j)
#         continue
#     pkl = os.path.join(to_dir + "Pickle/", mm + "_" + tmp + ".pickle")
#     shutil.copy(j, pkl)

