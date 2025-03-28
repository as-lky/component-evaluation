import torch
import os
import re
import gurobipy as gp
import pyscipopt
#import cplex
from typing import Type, cast, Self
from .mod import Component, Graphencode2Predict, Predict2Modify, PScores, Cansol

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
        self.time_limit = kwargs.get("time_limit", 10)
        ... # tackle parameters
    
    def work(self, input: Graphencode2Predict) -> Cansol:    
        
        self.begin()
        
        cansol = {}
        
        model = gp.read(self.instance)
        model.setParam('TimeLimit', self.time_limit)
        model.optimize()
        for var in model.getVars():
            cansol[var.VarName] = var.X
        
        self.end()
        return Cansol(model.ObjVal, cansol, model.MIPGap)
    
class SCIP(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 10)
        ... # tackle parameters
    
    def work(self, input: Graphencode2Predict) -> Cansol:    
        
        self.begin()
    
        solver = pyscipopt.Model()
        solver.readProblem(self.instance)
        solver.setRealParam('limits/time', self.time_limit)
        solver.optimize()
    
        cansol = {}
        
        for var in solver.getVars():
            cansol[var.name] = solver.getVal(var)

        self.end()
        return Cansol(solver.getObjVal(), cansol, solver.getGap())
    
class CPLEX(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit", 10)
        ... # tackle parameters
    
    def work(self, input: Graphencode2Predict) -> Cansol:    
        
        self.begin()
        
        cansol = {}
        
        model = cplex.Cplex()
        model.read(self.instance)
        model.parameters.timelimit.set(self.time_limit)
        model.solve()
        
        for var_name, var_value in zip(model.variables.get_names(), model.solution.get_values()):
            cansol[var_name] = var_value

        self.end()
        return Cansol(model.solution.get_objective_value(), cansol, model.solution.MIP.get_mip_relative_gap())


class GCN(Predict):
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        if "train_data_dir" in kwargs:
            self.train_data_dir = kwargs["train_data_dir"]
        ... # tackle parameters 
 
    def work(self, input: Graphencode2Predict) -> PScores:    
        
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
        # 模型训练不需要以sequence_name做路径 因为其与其他部分无关 只保留instance_name可以确保可复用性
        if os.path.exists(f'./Model/{self.taskname}/{instance_name}/GCN_predict.pth'): # TODO : add parameter for model name
            pathstr = f'./Model/{self.taskname}/{instance_name}/GCN_predict.pth'
        else :
            if not os.path.isdir(f'./logs/{self.taskname}/{instance_name}/'):
                os.mkdir(f'./logs/{self.taskname}/{instance_name}/')
            if not os.path.isdir(f'./logs/{self.taskname}/{instance_name}/train/'): # TODO : add parameter for model name
                os.mkdir(f'./logs/{self.taskname}/{instance_name}/train/')
            
        policy = GNNPolicy().to(DEVICE)
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
        return PScores(input.b_vars, scores)