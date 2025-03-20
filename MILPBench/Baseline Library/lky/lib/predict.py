import torch
import os
from typing import Type
from mod import Component, Graphencode2Predict, Predict2Modify

class Predict(Component):
    def __new__(cls, component, *args, **kwargs):
        if component == "gcn":
            return super().__new__(GCN, *args, **kwargs)
        else :
            raise ValueError("Predict component type is not defined")

class GCN(Predict):
    def __init__(self, component, *args, **kwargs):
        ... # tackle parameters
        
    def work(self, input: Type[Graphencode2Predict]) -> Type[Predict2Modify]:        
        # first check the model, if there is not then train using train instances
        if input.taskname =="IP":
            #Add position embedding for IP model, due to the strong symmetry
            from help.GCN import GNNPolicy_position as GNNPolicy
        else:
            from help.GCN import GNNPolicy
            
        model_name=f'{TaskName}.pth' # TODO : add parameter for model name
        pathstr = f'./models/{model_name}'
        policy = GNNPolicy().to(DEVICE)
        state = torch.load(pathstr, map_location=torch.device('cuda:0'))
        policy.load_state_dict(state)
        
        BD = policy(
            constraint_features.to(DEVICE),
            edge_indices.to(DEVICE),
            edge_features.to(DEVICE),
            variable_features.to(DEVICE),
        ).sigmoid().cpu().squeeze()

        #align the variable name betweend the output and the solver
        all_varname=[]
        for name in v_map:
            all_varname.append(name)
        binary_name=[all_varname[i] for i in b_vars]
        scores=[]#get a list of (index, VariableName, Prob, -1, type)
        for i in range(len(v_map)):
            type="C"
            if all_varname[i] in binary_name:
                type='BINARY'
            scores.append([i, all_varname[i], BD[i].item(), -1, type])