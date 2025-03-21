import numpy as np
import torch 
from typing import Type
from mod import Component, Preprocess2Graphencode, Graphencode2Predict
from help.helper import get_a_new2
from help.GCN import postion_get

class Graphencode(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "bi":
            return super().__new__(Bipartite, device, taskname, instance, sequence_name, *args, **kwargs)
        elif component == "tri":
            return super().__new__(Tripartite, device, taskname, instance, sequence_name, *args, **kwargs)
        else :
            raise ValueError("Graphencode component type is not defined")
 
    def __init__(self, device, taskname, instance, sequence_name):
        super().__init__(device, taskname, instance, sequence_name)
 
    def work(self, input: Type[Preprocess2Graphencode]) -> Type[Graphencode2Predict]:
        pass

class Bipartite(Graphencode):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)
        ... # tackle parameters
        
    def work(self, input: Type[Preprocess2Graphencode]) -> Type[Graphencode2Predict]:
        
        self.begin()
        
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(self.instance)
        constraint_features = c_nodes.cpu()
        constraint_features[np.isnan(constraint_features)] = 1  # remove nan value
        variable_features = v_nodes
        if self.taskname == "IP":
            variable_features = postion_get(variable_features) # position ? 
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        self.end()
        
        return Graphencode2Predict( constraint_features, edge_indices, edge_features, variable_features,
            v_map, v_nodes, c_nodes, b_vars )

class Tripartite(Graphencode):
    # TODO : get tripartite graph code
    
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)
        ... # tackle parameters
        
    def work(self, input: Type[Preprocess2Graphencode]) -> Type[Graphencode2Predict]:
        
        self.begin()
        
        A, v_map, v_nodes, c_nodes, b_vars = get_a_new2(self.instance)
        constraint_features = c_nodes.cpu()
        constraint_features[np.isnan(constraint_features)] = 1  # remove nan value
        variable_features = v_nodes
        if self.taskname == "IP":
            variable_features = postion_get(variable_features) # position ? 
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        edge_features = torch.ones(edge_features.shape)

        self.end()
        
        return Graphencode2Predict( constraint_features, edge_indices, edge_features, variable_features,
            v_map, v_nodes, c_nodes, b_vars )