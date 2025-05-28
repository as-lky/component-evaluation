import torch
import os
import re
import gurobipy as gp
import numpy as np
import random
import argparse
import time
import pickle
import torch_geometric
import copy
from gurobipy import *

parser = argparse.ArgumentParser(description="receive select instruction from higher level")
parser.add_argument("--device", required=True, choices=["cpu", "cuda", "cuda:2", "cuda:1", "cuda:3"], help="cpu or cuda")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "MIKS", "SC", "MIKSC"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance path")
parser.add_argument("--whole_time_limit", type=int, help="time limit for whole process")
parser.add_argument("--model_path", type=str, help="model path")
parser.add_argument("--search_LIH_MIH_NALNS_choose", type=float, help="LIH / MIH / NALNS choose parameter")

args = parser.parse_args()

device = args.device
instance = args.instance_path
taskname = args.taskname
model_path = args.model_path

# the convolution structure to convey information between the nodes in a bipartite graph
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output

# the tripartite graph convolution structure can be constructed using the bipartite graph convolution structure
# so there is no TripartiteGraphConvolution class
class GNNPolicy(torch.nn.Module):
    def __init__(self, random_feature=False, tripartite=False):
        super().__init__()
        self.tripartite = tripartite
        emb_size = 64
        cons_nfeats = 3 if random_feature else 2
        edge_nfeats = 1
        var_nfeats = 7 if random_feature else 6
        
        if tripartite:
            obj_nfeats = 5 if random_feature else 4

            # OBJ EMBEDDING
            self.obj_embedding = torch.nn.Sequential(
                torch.nn.LayerNorm(obj_nfeats),
                torch.nn.Linear(obj_nfeats, emb_size),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_size, emb_size),
                torch.nn.ReLU(),
            )

            # EDGE1 EMBEDDING
            self.edge_embedding1 = torch.nn.Sequential(
                torch.nn.LayerNorm(edge_nfeats),
            )
            
            # EDGE2 EMBEDDING
            self.edge_embedding2 = torch.nn.Sequential(
                torch.nn.LayerNorm(edge_nfeats),
            )


        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        if tripartite:
            self.conv_c_to_o = BipartiteGraphConvolution()
            self.conv_o_to_c = BipartiteGraphConvolution()
            self.conv_v_to_o = BipartiteGraphConvolution()
            self.conv_o_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
            #torch.nn.Sigmoid()
        )

        self.output_select = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
            #torch.nn.Sigmoid()
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features, obj_features=None, obj_variable_val=None, obj_constraint_val=None, edge_obj_var=None, edge_obj_cons=None
    ):
        if self.tripartite:
            reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

            # First step: linear embedding layers to a common dimension (64)
            constraint_features = self.cons_embedding(constraint_features)
            edge_features = self.edge_embedding(edge_features)
            variable_features = self.var_embedding(variable_features)
            obj_features = self.obj_embedding(obj_features)
            obj_variable_val = self.edge_embedding1(obj_variable_val)
            obj_constraint_val = self.edge_embedding2(obj_constraint_val)


            edge_var_obj = torch.stack([edge_obj_var[1], edge_obj_var[0]], dim=0)
            edge_cons_obj = torch.stack([edge_obj_cons[1], edge_obj_cons[0]], dim=0)

            # Two half convolutions
            for i in range(3):
                obj_features = self.conv_v_to_o(variable_features, edge_var_obj, obj_variable_val, obj_features)

                
                constraint_features = self.conv_v_to_c(
                    variable_features, reversed_edge_indices, edge_features, constraint_features
                    )
                
                constraint_features = self.conv_o_to_c(
                    obj_features, edge_obj_cons, obj_constraint_val, constraint_features
                )
                
                obj_features = self.conv_c_to_o(
                    constraint_features, edge_cons_obj, obj_constraint_val, obj_features
                )
                
                variable_features = self.conv_c_to_v(
                    constraint_features, edge_indices, edge_features, variable_features
                )
                
                variable_features = self.conv_o_to_v(
                    obj_features, edge_obj_var, obj_variable_val, variable_features
                )
            # A final MLP on the variable features
            # print(variable_features.shape)
            output = self.output_module(variable_features).squeeze(-1)
            select = self.output_select(variable_features).squeeze(-1)
            return output, select
        else:
            reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

            # First step: linear embedding layers to a common dimension (64)
            constraint_features = self.cons_embedding(constraint_features)
            edge_features = self.edge_embedding(edge_features)
            variable_features = self.var_embedding(variable_features)

            # Two half convolutions
            for i in range(3):
                
                constraint_features = self.conv_v_to_c(
                    variable_features, reversed_edge_indices, edge_features, constraint_features
                    )
                
                variable_features = self.conv_c_to_v(
                    constraint_features, edge_indices, edge_features, variable_features
                )
                
            # A final MLP on the variable features
            # print(variable_features.shape)
            output = self.output_module(variable_features).squeeze(-1)
            select = self.output_select(variable_features).squeeze(-1)
            return output, select

# bipartite graph data
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        assignment
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.assignment = assignment

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# tripartite graph data
class TripartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node tripartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        obj_features,
        obj_variable_val,
        obj_constraint_val,
        edge_obj_var,
        edge_obj_con,
        assignment
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.obj_features = obj_features
        self.obj_variable_val = obj_variable_val
        self.obj_constraint_val = obj_constraint_val
        self.edge_obj_var = edge_obj_var
        self.edge_obj_con = edge_obj_con
        self.assignment = assignment

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "edge_obj_var":
            return torch.tensor(
                [[1], [self.variable_features.size(0)]]
            )
        elif key == "edge_obj_con":
            return torch.tensor(
                [[1], [self.constraint_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# graph dataset
class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, tripartite=False):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.tripartite = tripartite

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        if not self.tripartite:
            with open(self.sample_files[index], "rb") as f:
                [variable_features, constraint_features, edge_indices, edge_features, solution] = pickle.load(f)

            graph = BipartiteNodeData(
                torch.FloatTensor(constraint_features),
                torch.LongTensor(edge_indices),
                torch.FloatTensor(edge_features),
                torch.FloatTensor(variable_features),
                torch.FloatTensor(solution)
            )

            # We must tell pytorch geometric how many nodes there are, for indexing purposes
            graph.num_nodes = len(constraint_features) + len(variable_features)
            graph.cons_nodes = len(constraint_features)
            graph.vars_nodes = len(variable_features)

            return graph
        else:
            with open(self.sample_files[index], "rb") as f:
                [variable_features, constraint_features, edge_indices, edge_features, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con, solution] = pickle.load(f)    

            graph = TripartiteNodeData(
                torch.FloatTensor(constraint_features),
                torch.LongTensor(edge_indices),
                torch.FloatTensor(edge_features),
                torch.FloatTensor(variable_features),
                torch.FloatTensor(obj_features),
                torch.FloatTensor(obj_variable_val),
                torch.FloatTensor(obj_constraint_val),
                torch.LongTensor(edge_obj_var),
                torch.LongTensor(edge_obj_con),
                torch.FloatTensor(solution)
            )

            # We must tell pytorch geometric how many nodes there are, for indexing purposes
            graph.num_nodes = len(constraint_features) + len(variable_features) + 1
            graph.cons_nodes = len(constraint_features)
            graph.vars_nodes = len(variable_features)
            graph.obj_nodes = 1

            return graph

# bipartite graph data
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        assignment
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.assignment = assignment

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# tripartite graph data
class TripartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node tripartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        obj_features,
        obj_variable_val,
        obj_constraint_val,
        edge_obj_var,
        edge_obj_con,
        assignment
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.obj_features = obj_features
        self.obj_variable_val = obj_variable_val
        self.obj_constraint_val = obj_constraint_val
        self.edge_obj_var = edge_obj_var
        self.edge_obj_con = edge_obj_con
        self.assignment = assignment

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "edge_obj_var":
            return torch.tensor(
                [[1], [self.variable_features.size(0)]]
            )
        elif key == "edge_obj_con":
            return torch.tensor(
                [[1], [self.constraint_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# graph dataset
class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, tripartite=False):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.tripartite = tripartite

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        if not self.tripartite:
            with open(self.sample_files[index], "rb") as f:
                [variable_features, constraint_features, edge_indices, edge_features, solution] = pickle.load(f)

            graph = BipartiteNodeData(
                torch.FloatTensor(constraint_features),
                torch.LongTensor(edge_indices),
                torch.FloatTensor(edge_features),
                torch.FloatTensor(variable_features),
                torch.FloatTensor(solution)
            )

            # We must tell pytorch geometric how many nodes there are, for indexing purposes
            graph.num_nodes = len(constraint_features) + len(variable_features)
            graph.cons_nodes = len(constraint_features)
            graph.vars_nodes = len(variable_features)

            return graph
        else:
            with open(self.sample_files[index], "rb") as f:
                [variable_features, constraint_features, edge_indices, edge_features, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con, solution] = pickle.load(f)    

            graph = TripartiteNodeData(
                torch.FloatTensor(constraint_features),
                torch.LongTensor(edge_indices),
                torch.FloatTensor(edge_features),
                torch.FloatTensor(variable_features),
                torch.FloatTensor(obj_features),
                torch.FloatTensor(obj_variable_val),
                torch.FloatTensor(obj_constraint_val),
                torch.LongTensor(edge_obj_var),
                torch.LongTensor(edge_obj_con),
                torch.FloatTensor(solution)
            )

            # We must tell pytorch geometric how many nodes there are, for indexing purposes
            graph.num_nodes = len(constraint_features) + len(variable_features) + 1
            graph.cons_nodes = len(constraint_features)
            graph.vars_nodes = len(variable_features)
            graph.obj_nodes = 1

            return graph

# function to read the *.lp file and return the basic information of the instance and the encoded bipartite graph features
# random_feature is true when encoding the instance into a graph with random features
def get_a_new2(instance, random_feature = False):
    model = gp.read(instance)
    value_to_num = {}
    num_to_value = {}
    value_to_type = {}
    value_num = 0
    #N represents the number of decision variables
    #M represents the number of constraints
    #K [i] represents the number of decision variables in the i-th constraint
    #Site [i] [j] represents which decision variable is the jth decision variable of the i-th constraint
    #Value [i] [j] represents the coefficient of the jth decision variable of the i-th constraint
    #Constraint [i] represents the number to the right of the i-th constraint
    #Constrict_type [i] represents the type of the i-th constraint, 1 represents<, 2 represents>, and 3 represents=
    #Coefficient [i] represents the coefficient of the i-th decision variable in the objective function
    n = model.NumVars
    m = model.NumConstrs
    k = []
    site = []
    value = []
    constraint = []
    constraint_type = []
    for cnstr in model.getConstrs():
        if(cnstr.Sense == '<'):
            constraint_type.append(1)
        elif(cnstr.Sense == '>'):
            constraint_type.append(2) 
        else:
            constraint_type.append(3) 
        
        constraint.append(cnstr.RHS)

        now_site = []
        now_value = []
        row = model.getRow(cnstr)
        k.append(row.size())
        for i in range(row.size()):
            if(row.getVar(i).VarName not in value_to_num.keys()):
                value_to_num[row.getVar(i).VarName] = value_num
                num_to_value[value_num] = row.getVar(i).VarName
                value_num += 1
            now_site.append(value_to_num[row.getVar(i).VarName])
            now_value.append(row.getCoeff(i))
        site.append(now_site)
        value.append(now_value)

    coefficient = {}
    lower_bound = {}
    upper_bound = {}
    value_type = {}
    for val in model.getVars():
        if(val.VarName not in value_to_num.keys()):
            value_to_num[val.VarName] = value_num
            num_to_value[value_num] = val.VarName
            value_num += 1
        coefficient[value_to_num[val.VarName]] = val.Obj
        lower_bound[value_to_num[val.VarName]] = val.LB
        upper_bound[value_to_num[val.VarName]] = val.UB
        value_type[value_to_num[val.VarName]] = val.Vtype

    #1 minimize, -1 maximize
    obj_type = model.ModelSense
    
    variable_features = []
    constraint_features = []
    edge_indices = [[], []] 
    edge_features = []

    for i in range(n):
        now_variable_features = []
        now_variable_features.append(coefficient[i])
        if(lower_bound[i] == float("-inf")):
            now_variable_features.append(0)
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
            now_variable_features.append(lower_bound[i])
        if(upper_bound[i] == float("inf")):
            now_variable_features.append(0)
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
            now_variable_features.append(upper_bound[i])
        if(value_type[i] == 'C'):
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
        if random_feature:
            now_variable_features.append(random.random())
        variable_features.append(now_variable_features)
    
    for i in range(m):
        now_constraint_features = []
        now_constraint_features.append(constraint[i])
        now_constraint_features.append(constraint_type[i])
        if random_feature:
            now_constraint_features.append(random.random())
        constraint_features.append(now_constraint_features)
    
    for i in range(m):
        for j in range(k[i]):
            edge_indices[0].append(i)
            edge_indices[1].append(site[i][j])
            edge_features.append([value[i][j]])

    return constraint_features, edge_indices, edge_features, variable_features, num_to_value, n

# function to read the *.lp file and return the basic information of the instance and the encoded tripartite graph features
# random_feature is true when encoding the instance into a graph with random features
def get_a_new3(instance, random_feature = False):
    model = gp.read(instance)
    value_to_num = {}
    num_to_value = {}
    value_to_type = {}
    value_num = 0
    #N represents the number of decision variables
    #M represents the number of constraints
    #K [i] represents the number of decision variables in the i-th constraint
    #Site [i] [j] represents which decision variable is the jth decision variable of the i-th constraint
    #Value [i] [j] represents the coefficient of the jth decision variable of the i-th constraint
    #Constraint [i] represents the number to the right of the i-th constraint
    #Constrict_type [i] represents the type of the i-th constraint, 1 represents<, 2 represents>, and 3 represents=
    #Coefficient [i] represents the coefficient of the i-th decision variable in the objective function
    n = model.NumVars
    m = model.NumConstrs
    k = []
    site = []
    value = []
    constraint = []
    constraint_type = []
    for cnstr in model.getConstrs():
        if(cnstr.Sense == '<'):
            constraint_type.append(1)
        elif(cnstr.Sense == '>'):
            constraint_type.append(2) 
        else:
            constraint_type.append(3) 
        
        constraint.append(cnstr.RHS)

        now_site = []
        now_value = []
        row = model.getRow(cnstr)
        k.append(row.size())
        for i in range(row.size()):
            if(row.getVar(i).VarName not in value_to_num.keys()):
                value_to_num[row.getVar(i).VarName] = value_num
                num_to_value[value_num] = row.getVar(i).VarName
                value_num += 1
            now_site.append(value_to_num[row.getVar(i).VarName])
            now_value.append(row.getCoeff(i))
        site.append(now_site)
        value.append(now_value)

    coefficient = {}
    lower_bound = {}
    upper_bound = {}
    value_type = {}
    for val in model.getVars():
        if(val.VarName not in value_to_num.keys()):
            value_to_num[val.VarName] = value_num
            num_to_value[value_num] = val.VarName
            value_num += 1
        coefficient[value_to_num[val.VarName]] = val.Obj
        lower_bound[value_to_num[val.VarName]] = val.LB
        upper_bound[value_to_num[val.VarName]] = val.UB
        value_type[value_to_num[val.VarName]] = val.Vtype

    #1 minimize, -1 maximize
    obj_type = model.ModelSense
    
    edge_obj_var = [[0] * n, [i for i in range(n)]]
    edge_obj_con = [[0] * m, [i for i in range(m)]]
    obj_variable_val = []
    obj_constraint_val = []
    obj_features = [[]]
    
    variable_features = []
    constraint_features = []
    edge_indices = [[], []] 
    edge_features = []

    cnt = 0
    MAX, MIN = -2e9, 2e9
    for i in range(n):
        obj_variable_val.append([coefficient[i]])
        if coefficient[i] != 0:
            cnt += 1
        MAX = max(MAX, coefficient[i])
        MIN = min(MIN, coefficient[i])
    obj_features[0] = [obj_type, cnt, MAX, MIN]
    if random_feature:
        obj_features[0].append(random.random())
    for i in range(m):
        obj_constraint_val.append([constraint[i]])
    
    for i in range(n):
        now_variable_features = []
        now_variable_features.append(coefficient[i])
        if(lower_bound[i] == float("-inf")):
            now_variable_features.append(0)
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
            now_variable_features.append(lower_bound[i])
        if(upper_bound[i] == float("inf")):
            now_variable_features.append(0)
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
            now_variable_features.append(upper_bound[i])
        if(value_type[i] == 'C'):
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
        if random_feature:
            now_variable_features.append(random.random())
        variable_features.append(now_variable_features)
    
    for i in range(m):
        now_constraint_features = []
        now_constraint_features.append(constraint[i])
        now_constraint_features.append(constraint_type[i])
        if random_feature:
            now_constraint_features.append(random.random())
        constraint_features.append(now_constraint_features)
    
    for i in range(m):
        for j in range(k[i]):
            edge_indices[0].append(i)
            edge_indices[1].append(site[i][j])
            edge_features.append([value[i][j]])

    return constraint_features, edge_indices, edge_features, variable_features, num_to_value, n, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con

# using GCN to predict the solution of the instance
def predict():
    print("GCN predict...")

    DEVICE = device     
    instance_name = os.path.basename(instance)
    instance_name = re.match(r"(.*)_[0-9]+", instance_name)
    if instance_name == None:
        raise ValueError("instance name error!")
    else :
        instance_name = instance_name.group(1)

    tripartite = True 
    policy = GNNPolicy(random_feature=True, tripartite=tripartite).to(DEVICE)
    policy.load_state_dict(torch.load(model_path, policy.state_dict()))
    instance_name = os.path.basename(instance)
    instance_name = re.match(r"(.*_[0-9]+)\.lp", instance_name)
    instance_name = instance_name.group(1)


    W = f'./logs/'
    pk = os.path.join(W, instance_name) + '.pickle'

    # if the pickle file(containing the features) does not exist, then generate the features and save them
    if not os.path.exists(pk):
        constraint_features, edge_indices, edge_features, variable_features, num_to_value, n, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con = get_a_new3(instance, random_feature=True)
        sol = []
        with open(pk, "wb") as f:
            pickle.dump([variable_features, constraint_features, edge_indices, edge_features, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con, sol], f)
            
    # load the data using Dataset batch with a batch whose size is 1 
    file = [pk]
    data = GraphDataset(file, tripartite=tripartite)
    loader = torch_geometric.loader.DataLoader(data, batch_size = 1)

    logits, select = None, None
    for batch in loader:
        batch = batch.to(device)
        # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
        if not tripartite:
            logits, select = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
        else:
            logits, select = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                batch.obj_features,
                batch.obj_variable_val,
                batch.obj_constraint_val,
                batch.edge_obj_var,
                batch.edge_obj_con,
            )

    return logits, select

def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, now_col, lower_bound, upper_bound, value_type):
    '''
    Function Description:
    Use Gurobi solver to solve the problem based on the provided problem instance and current solution and current selection.

    Parameter description:
    -N: The number of decision variables in the problem instance.
    -M: The number of constraints for problem instances.
    -K: k [i] represents the number of decision variables for the i-th constraint.
    -Site: site [i] [j] represents which decision variable is the jth decision variable of the i-th constraint.
    -Value: value [i] [j] represents the coefficient of the jth decision variable of the i-th constraint.
    -Constraint: constraint [i] represents the number to the right of the i-th constraint.
    -Constrict_type: constrict_type [i] represents the type of the i-th constraint, 1 represents<=, 2 represents>=
    -Coefficient: coefficient [i] represents the coefficient of the i-th decision variable in the objective function.
    -Time_imit: Maximum solution time.
    -Obj_type: Is the problem a maximization problem or a minimization problem.
    -Now_sol: represents the current solution.
    -Now_col: represents the current selection of decision variables, 0 means selected, 1 means not selected.
    '''
    begin_time = time.time()
    model = Model("Gurobi")
    model.feasRelaxS(0,False,False,True)
    site_to_new = {}
    new_to_site = {}
    new_num = 0
    x = []
    for i in range(n):
        if(now_col[i] == 1):
            site_to_new[i] = new_num
            new_to_site[new_num] = i
            new_num += 1
            if(value_type[i] == 'B'):
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
            elif(value_type[i] == 'C'):
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
            else:
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))
                
    for i in range(m):
        constr = 0
        flag = 0
        for j in range(k[i]):
            if(now_col[site[i][j]] == 1):
                constr += x[site_to_new[site[i][j]]] * value[i][j]
                flag = 1
            else:
                constr += now_sol[site[i][j]] * value[i][j]

        if(flag == 1):
            if(constraint_type[i] == 1):
                model.addConstr(constr <= constraint[i])
            elif(constraint_type[i] == 2):
                model.addConstr(constr >= constraint[i])
            else:
                model.addConstr(constr == constraint[i])
        else:
            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    # No feasible solution
                    print("QwQ fine")
                    print(constr,  constraint[i])
                    return -1, -1, -1, -1
            else:
                if(constr < constraint[i]):
                    print("QwQ fine")
                    print(constr,  constraint[i])
                    return -1, -1, -1, -1
    
    coeff = 0
    flag = 0
    for i in range(n):
        if(now_col[i] == 1):
            coeff += x[site_to_new[i]] * coefficient[i]
            flag = 1
        else:
            coeff += now_sol[i] * coefficient[i]
    
    if flag == 1:
        if(obj_type == 'maximize'):
            model.setObjective(coeff, GRB.MAXIMIZE)
        else:
            model.setObjective(coeff, GRB.MINIMIZE)
                    
        model.setParam('SolutionLimit', 1)
        model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
        model.optimize()

    try:
        new_sol = []
        for i in range(n):
            if(now_col[i] == 0):
                new_sol.append(now_sol[i])
            else:
                if(value_type[i] == 'C'):
                    new_sol.append(x[site_to_new[i]].X)
                else:
                    new_sol.append((int)(x[site_to_new[i]].X))
        if model.NumVars == 0:
            return 1, new_sol, coeff, 0
        
        return 1, new_sol, model.ObjVal, model.MIPGap
    except:
        return -1, -1, -1, -1

# using Sr strategy to repair the infeasible solution
def repair(logits, select, time_limit):
    print("Sr repair...")

    constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value=get_a_new2(instance)

    time_limit = time_limit

    if type(input.select) == list:
        select = torch.tensor(np.array(input.select))
    else:
        select = input.select.clone()
    new_select = select.clone()
    new_select, _tmp = torch.sort(new_select)

    now_sol = input.logits
        
    for i in range(n):
        if(value_type[i] != 'C'):
            now_sol[i] = int(now_sol[i] + 0.5)
        now_sol[i] = min(now_sol[i], upper_bound[i])
        now_sol[i] = max(now_sol[i], lower_bound[i])
    
    result_pair = (0, 0, 0)
    add_flag = 0
    for turn2 in range(11):
        turn = 10 - turn2
        choose = []
        rate = (int)(0.1 * turn * (n - 1))
        for i in range(n):
            if(select[i] >= new_select[rate]):
                choose.append(1)
            else:
                choose.append(0)
        flag, sol, obj, gap = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, choose, lower_bound, upper_bound, value_type)
        if(flag == 1):
            add_flag = 1
            result_pair = (sol, obj, gap)
            break
        
    cansol = {}
    for i in range(n):
        cansol[num_to_value[i]] = result_pair[0][i]

    return result_pair[1], cansol, result_pair[2]

def split_problem(lp_file):
    """
    Function Description:
    Solve the given problem instance using the Gurobi solver.

    Parameter Description:
    - n: The number of decision variables in the problem instance.
    - m: The number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable is the j-th variable in the i-th constraint.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 indicates <= and 2 indicates >=.
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - time_limit: The maximum solving time.
    - obj_type: Indicates whether the problem is a maximization or minimization problem.
    """
    model = gp.read(lp_file)
    n = model.NumVars
    m = model.NumConstrs
    k = []
    site = []
    value = []
    constraint = []
    constraint_type = []
    coefficient = []
    obj_type = model.ModelSense
    upper_bound = []
    lower_bound = []
    value_type = []
    var_name_to_index = {}

    objective = model.getObjective()
    temp_coeff = []
    # using a dictionary to map variable names to indices, which is much faster than using a list
    temp_varname = {}
    for i in range(objective.size()):
        temp_coeff.append(objective.getCoeff(i))
        temp_varname[objective.getVar(i).VarName] = i

    i = 0
    for var in model.getVars():
        var_name_to_index[var.VarName] = i
        upper_bound.append(var.UB)
        lower_bound.append(var.LB)
        value_type.append(var.VType)
        if var.VarName not in temp_varname:
            coefficient.append(0)
        else:
            coefficient.append(temp_coeff[temp_varname[var.VarName]])
        i+=1

    for cnstr in model.getConstrs():
        if(cnstr.Sense == '<'):
            constraint_type.append(1)
        elif(cnstr.Sense == '>'):
            constraint_type.append(2) 
        else:
            constraint_type.append(3) 
        
        constraint.append(cnstr.RHS)

        now_site = []
        now_value = []
        row = model.getRow(cnstr)
        k.append(row.size())
        for i in range(row.size()):
            now_site.append(var_name_to_index[row.getVar(i).VarName])
            now_value.append(row.getCoeff(i))
        site.append(now_site)
        value.append(now_value)
        
    return n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type

def select_neighborhood(n, m, k, site, value, constraint, initial_solution, current_solution, objective_coefficient):
    # This new heuristic approach combines the principles of simulated annealing 
    # with the adaptive scoring of decision variables based on their contributions 
    # to violated constraints while incorporating randomness to enhance exploration 
    # of the solution space.
    
    # Initialize the neighbor_score array
    neighbor_score = np.zeros(n)

    # Calculate the current objective value
    current_objective_value = np.dot(current_solution, objective_coefficient)

    # Identify the contributions of each variable
    variable_contributions = np.zeros(n)

    # Loop through the constraints and identify violated ones
    for i in range(m):
        lhs_value = sum(value[i][j] * current_solution[site[i][j]] for j in range(k[i]))
        if lhs_value > constraint[i]:  # Only consider violated constraints
            for j in range(k[i]):
                var_index = site[i][j]
                # Calculate the contribution of violating variables
                variable_contributions[var_index] += (value[i][j] * (current_solution[var_index] == 1))

    # Score variables based on contributions and potential improvements
    for index in range(n):
        improvement = objective_coefficient[index] - variable_contributions[index]
        # Adaptively incorporate a factor based on current solution
        neighbor_score[index] = improvement + (current_solution[index] * 0.5)

    # Simulated annealing-inspired randomness for exploration
    temperature = np.random.uniform(0.1, 1.0)
    randomness = np.random.uniform(-temperature, temperature, size=n)
    neighbor_score += randomness

    return neighbor_score

# neighborhood adaptive large neighborhood search (NALNS) algorithm main function
def greedy_one(now_instance_data, time_limit, choose_=0.5):
    begin_time = time.time()
    set_time = time_limit
    n = now_instance_data[0]
    m = now_instance_data[1]
    k = now_instance_data[2]
    site = now_instance_data[3]
    value = now_instance_data[4] 
    constraint = now_instance_data[5]
    constraint_type = now_instance_data[6] 
    coefficient = now_instance_data[7]
    obj_type = now_instance_data[8]
    lower_bound = now_instance_data[9]
    upper_bound = now_instance_data[10]
    value_type = now_instance_data[11]
    initial_sol = now_instance_data[12]

    choose = choose_
    best_val = eval(n, coefficient, initial_sol)
    
    turn_time = [time.time() - begin_time]
    turn_ans = [best_val]
    turn_limit = 50
    GAP = 0
    now_sol = initial_sol
    while(time.time() - begin_time <= set_time):
        neighbor_score = select_neighborhood(
                            n, 
                            m, 
                            copy.deepcopy(k),
                            copy.deepcopy(site), 
                            copy.deepcopy(value), 
                            copy.deepcopy(constraint), 
                            copy.deepcopy(initial_sol), 
                            copy.deepcopy(now_sol), 
                            copy.deepcopy(coefficient)
                        )
        indices = np.argsort(neighbor_score)[::-1]
        color = np.zeros(n)
        for i in range(int(n * choose)):
            color[indices[i]] = 1
        new_sol, now_val, now_flag, now_gap = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(set_time - (time.time() - begin_time), turn_limit), obj_type, lower_bound, upper_bound, value_type, now_sol, color)
        if(now_flag == -1):
            continue
        GAP = now_gap
        #Maximize
        if(obj_type == -1):
            if(now_val > best_val):
                now_sol = new_sol
                best_val = now_val
        else:
            if(now_val < best_val):
                now_sol = new_sol
                best_val = now_val

        turn_ans.append(best_val) 
        turn_time.append(time.time() - begin_time)
        print(turn_ans[-1], turn_time[-1])
    return(turn_ans, turn_time, GAP)

# using NALNS strategy to search better solutions
def search(objval, cansol, gap, time_limit, choose, result_list):
    print("NALNS search...")
    n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type = split_problem(instance)

    new_sol = []
    tmp = gp.read(instance)
    for var in tmp.getVars():
        new_sol.append(input.cansol[var.VarName])
    
    time_limit = time_limit

    new_site = []
    new_value = []
    new_constraint = np.zeros(m)
    new_constraint_type = np.zeros(m, int)
    for i in range(m):
        new_site.append(np.zeros(k[i], int))
        new_value.append(np.zeros(k[i]))
        for j in range(k[i]):
            new_site[i][j] = site[i][j]
            new_value[i][j] = value[i][j]
        new_constraint[i] = constraint[i]
        new_constraint_type[i] = constraint_type[i]
    
    new_coefficient = np.zeros(n)
    new_lower_bound = np.zeros(n)
    new_upper_bound = np.zeros(n)
    new_value_type = np.zeros(n, int)
    new_new_sol = np.zeros(n)
    for i in range(n):
        new_coefficient[i] = coefficient[i]
        new_lower_bound[i] = lower_bound[i]
        new_upper_bound[i] = upper_bound[i]
        if(value_type[i] == 'B'):
            new_value_type[i] = 0
        elif(value_type[i] == 'C'):
            new_value_type[i] = 1
        else:
            new_value_type[i] = 2
        new_new_sol[i] = new_sol[i]

    now_instance = (n, m, k, new_site, new_value, new_constraint, new_constraint_type, new_coefficient, obj_type, new_lower_bound, new_upper_bound, new_value_type, new_new_sol)
    now_sol, now_time, now_gap = greedy_one(now_instance, time_limit, choose)

    for _ in range(len(now_sol)):
        result_list.append((result_list[0][0] + now_time[_], now_sol[_]))
    
result_list_obj_time = []
start_time = time.time()
whole_time_limit = args.whole_time_limit

# using gcn for predict(tripartite graph with random feature), Sr for repair and NALNS for search
# get the result list of objective value and time
predict_, select = predict()
objval, cansol, gap = repair(predict_, select, whole_time_limit - (time.time() - start_time))
result_list_obj_time.append((time.time() - start_time, objval))    
search(objval, cansol, gap, whole_time_limit - (time.time() - start_time), args.search_ACP_LNS_block, args.search_ACP_LNS_max_turn_ratio, result_list_obj_time)
print(result_list_obj_time)
    