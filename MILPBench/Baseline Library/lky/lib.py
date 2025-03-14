import numpy as np
from GCN import postion_get
from helper import get_a_new2
import torch 

# component parameters

class Component:
    def __init__(self):
        pass

    def work(
        self, input, *args
    ):  # args are the component select parameters, which should be interpreted and correspended to the input
        output = 1
        return output


class Graphencode(Component):
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
    
    def getnew3(self, task_type, ins_name_to_read):
        ... # TODO : get tripartite graph code

    def work(self, input, *args):
        ... #
        
class Predict(Component): # 提供实例
    def GNNPredict(self, input, *args): # 接受问题实例
        # train
        # infer
        ...    
    def work(self, input, *args):
        output = 2
        return output


class Predict_Modify_Search(Component):
    
    def add_radius_constraint(self): # 接受predict结果
        
        ...
    def work(self, input, *args):
    
        
class Search(Component):
    def Solver(self) : # TODO : check 其接受什么东西
    def work(self, input, *args): 
        output = 3
        return output


r, w = Search(), Predict()
print("ASDD")
print(r.work(1), w.work(2))
