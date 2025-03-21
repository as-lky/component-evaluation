import torch
import os
from typing import Type
from mod import Component, Graphencode2Predict, Predict2Modify, Scores

class Predict(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "gcn":
            return super().__new__(GCN, device, taskname, instance, sequence_name, *args, **kwargs)
        else :
            raise ValueError("Predict component type is not defined")

    def __init__(self, device, taskname, instance, sequence_name):
        super().__init__(device, taskname, instance, sequence_name)

    def work(self, input: Type[Graphencode2Predict]) -> Type[Predict2Modify]:        
        pass

class GCN(Predict):
        
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)
        ... # tackle parameters 
 
    def work(self, input: Type[Graphencode2Predict]) -> Type[Scores]:    
        
        self.begin()
        # first check the model, if there is not then train using train instances
        if self.taskname == "IP":
            #Add position embedding for IP model, due to the strong symmetry
            from help.GCN import GNNPolicy_position as GNNPolicy
        else:
            from help.GCN import GNNPolicy
        
        DEVICE = self.device     
        
        if os.path.exists(f'./Model/{self.taskname}/GCN_predict.pth'): # TODO : add parameter for model name
            pathstr = f'./Model/{self.taskname}/GCN_predict.pth'
        else :
            ...
            # LP instance dir TRAIN train dir
            # train! TODO
        policy = GNNPolicy().to(DEVICE)
        state = torch.load(pathstr, map_location=torch.device('cuda:0')) # why cuda?
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
        return Scores(input.b_vars, scores)