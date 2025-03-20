import numpy as np
import torch 
from typing import Type
from mod import Component, Preprocess2Graphencode, Graphencode2Predict
from help.helper import get_a_new2
from help.GCN import postion_get

class Graphencode(Component):
    def __new__(cls, component, *args, **kwargs):
        if component == "bi":
            return super().__new__(Bipartite, *args, **kwargs)
        elif component == "tri":
            return super().__new__(Tripartite, *args, **kwargs)
        else :
            raise ValueError("Graphencode component type is not defined")

class Bipartite(Graphencode):
    def __init__(self, component, *args, **kwargs):
        ... # tackle parameters
        
    def work(self, input: Type[Preprocess2Graphencode]) -> Type[Graphencode2Predict]:
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(input.instance)
        constraint_features = c_nodes.cpu()
        constraint_features[np.isnan(constraint_features)] = 1  # remove nan value
        variable_features = v_nodes
        if input.taskname == "IP":
            variable_features = postion_get(variable_features) # position ? 
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)
        return Graphencode2Predict(input.device, input.taskname, input.instance, 
            constraint_features, edge_indices, edge_features, variable_features,
            v_map, v_nodes, c_nodes, b_vars
        )

class Tripartite(Graphencode):
    # TODO : get tripartite graph code
    def __init__(self, component, *args, **kwargs):
        ... # tackle parameters
            
    def getnew2(self, task_type, ins_name_to_read):  # TODO : check the type of ins_name_to_read
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(ins_name_to_read)
        constraint_features = c_nodes.cpu()
        constraint_features[np.isnan(constraint_features)] = 1  # remove nan value
        variable_features = v_nodes
        if task_type == "IP":
            variable_features = postion_get(variable_features) # position ? 
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)
        return constraint_features, edge_indices, edge_features, variable_features
    
    def work(self, input: Type[Preprocess2Graphencode]) -> Type[Graphencode2Predict]:
        return Graphencode2Predict()