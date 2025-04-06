import torch
import os
import re
import gurobipy as gp
import pyscipopt
import subprocess
#import cplex
from typing import Type, cast, Self
from .mod import Component, Graphencode2Predict, Predict2Modify, Cantsol, Cansol2M

class Predict(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "gcn":
            cls = GCN
        elif component == "gurobi":
            cls = Gurobi 
        elif component == "scip":
            cls = SCIP
        elif component == "cplex":
            cls = CPLEX
        else:
            raise ValueError("Predict component type is not defined")
        
        return super().__new__( cast(type[Self], cls) )

    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)


    def work(self, input: Graphencode2Predict) -> Predict2Modify:...

class Gurobi(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        ... # tackle parameters
    
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
        
        cansol = {}
        
        model = gp.read(self.instance)
        model.setParam('TimeLimit', self.time_limit)
        model.optimize()
        for var in model.getVars():
            cansol[var.VarName] = var.X
        
        self.end()
        return Cansol2M(model.ObjVal, cansol, model.MIPGap)
    
class SCIP(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        ... # tackle parameters
    
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
    
        solver = pyscipopt.Model()
        solver.readProblem(self.instance)
        solver.setRealParam('limits/time', self.time_limit)
        solver.optimize()
    
        cansol = {}
        
        for var in solver.getVars():
            cansol[var.name] = solver.getVal(var)

        self.end()
        return Cansol2M(solver.getObjVal(), cansol, solver.getGap())
    
class CPLEX(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 10)
        ... # tackle parameters
    
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
        
        cansol = {}
        
        model = cplex.Cplex()
        model.read(self.instance)
        model.parameters.timelimit.set(self.time_limit)
        model.solve()
        
        for var_name, var_value in zip(model.variables.get_names(), model.solution.get_values()):
            cansol[var_name] = var_value

        self.end()
        return Cansol2M(model.solution.get_objective_value(), cansol, model.solution.MIP.get_mip_relative_gap())


class GCN(Predict):
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        if "train_data_dir" in kwargs:
            self.train_data_dir = kwargs["train_data_dir"]
        ... # tackle parameters 
 
    def work(self, input: Graphencode2Predict) -> Cantsol:    
        
        self.begin()
        # first check the model, if there is not then train using train instances
        if self.taskname == "IP":
            #Add position embedding for IP model, due to the strong symmetry
            from .help.GCN.GCN import GNNPolicy_position as GNNPolicy
        else:
            from .help.GCN.GCN import GNNPolicy
        
        DEVICE = self.device     
        
        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*)_[0-9]+", instance_name)
        if instance_name == None:
            raise ValueError("instance name error!")
        else :
            instance_name = instance_name.group(1)
            
        pathstr = ""
        # 模型训练不需要以sequence_name做路径 因为其与其他部分无关 只保留instance_name 和参数可以确保可复用性 ×
        model_dir = f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'
        model_path = f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/model_best.pth'
        if os.path.exists(model_path): # TODO : add parameter for model name
            pathstr = model_path
        else :
            if not os.path.isdir('./logs/'):
                os.mkdir('./logs')
            if not os.path.isdir(f'./logs/train/'):
                os.mkdir('./logs/train')
            if not os.path.isdir(f'./logs/train/{self.taskname}/'):
                os.mkdir(f'./logs/train/{self.taskname}')
            if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/'):
                os.mkdir(f'./logs/train/{self.taskname}/{instance_name}')
            if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
                os.mkdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
            if not os.path.isdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'):
                os.mkdir(f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')
            
            if not os.path.isdir('./Model/'):
                os.mkdir('./Model/')
            if not os.path.isdir(f'./Model/{self.taskname}'):
                os.mkdir(f'./Model/{self.taskname}')
            if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}'):
                os.mkdir(f'./Model/{self.taskname}/{instance_name}')
            if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
                os.mkdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
            if not os.path.isdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}'):
                os.mkdir(f'./Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')
            
            W = f'./logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'
            if self.sequence_name[0][-1] == 'r':
                subprocess.run(["python", "lib/help/GCN/trainPredictModel.py", "--device", f"{self.device}", "--taskname", f"{self.taskname}", "--train_data_dir", f"{self.train_data_dir}",
                                "--log_dir", f"{W}", "--model_save_dir", f"{model_dir}", "--random_feature"])    
            else:
                subprocess.run(["python", "lib/help/GCN/trainPredictModel.py", "--device", f"{self.device}", "--taskname", f"{self.taskname}", "--train_data_dir", f"{self.train_data_dir}",
                                "--log_dir", f"{W}", "--model_save_dir", f"{model_dir}"])    
            pathstr = model_path
            # train_data_dir + LP / Pickle    
            
        policy = GNNPolicy(random_feature=True if self.sequence_name[0][-1] == 'r' else False).to(DEVICE)
        state = torch.load(pathstr, map_location=torch.device(DEVICE)) # TODO: check why cuda?
        policy.load_state_dict(state)
        
        BD = policy(
            input.constraint_features.to(DEVICE),
            input.edge_indices.to(DEVICE),
            input.edge_features.to(DEVICE),
            input.variable_features.to(DEVICE),
        ).sigmoid().cpu().squeeze()

        # align the variable name betweend the output and the solver
        all_varname=[]
        for name in input.v_map:
            all_varname.append(name)
        scores=[] # get a list of (index, VariableName, Prob)
        for i in range(len(input.v_map)):
            scores.append([i, all_varname[i], BD[i].item()])
        self.end()
        return Cantsol(scores)