import gurobipy as gp
#instance = './Dataset/MIKS_medium_instance/MIKS_medium_instance/LP/MIKS_medium_instance_4.lp'
instance = './Dataset/MIKS_easy_instance/MIKS_easy_instance/LP/MIKS_easy_instance_0.lp'
model = gp.read(instance)
model.setParam('TimeLimit', 300)
model.optimize()