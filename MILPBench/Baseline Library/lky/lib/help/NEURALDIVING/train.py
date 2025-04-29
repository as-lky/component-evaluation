import argparse
import pickle
from pathlib import Path
from typing import Union
import re
import os
import torch
import torch.nn.functional as F
import torch_geometric
import gurobipy as gp
import random
from pytorch_metric_learning import losses

from graphcnn import GNNPolicy

__all__ = ["train"]

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


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, log_dir, random_feature, tripartite):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.log_dir = log_dir
        self.random_feature = random_feature
        self.tripartite = tripartite

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        if not self.tripartite:
            instance, solution_path = self.sample_files[index]
            instance_name = os.path.basename(instance)
            instance_name = re.match(r"(.*_[0-9]+)\.lp", instance_name)
            instance_name = instance_name.group(1)
            pk = os.path.join(self.log_dir, instance_name) + '.pickle'
            if os.path.exists(pk):
                with open(pk, "rb") as f:
                    [variable_features, constraint_features, edge_indices, edge_features, solution] = pickle.load(f)    
            else:
                constraint_features, edge_indices, edge_features, variable_features, num_to_value, n = get_a_new2(instance, self.random_feature)
                
                with open(solution_path, "rb") as f:
                    solution = pickle.load(f)[0]
                sol = []
                for i in range(n):
                    sol.append(solution[num_to_value[i]])
                    
                with open(pk, "wb") as f:
                    pickle.dump([variable_features, constraint_features, edge_indices, edge_features, sol], f)
                solution = sol
                    
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
            instance, solution_path = self.sample_files[index]
            instance_name = os.path.basename(instance)
            instance_name = re.match(r"(.*_[0-9]+)\.lp", instance_name)
            instance_name = instance_name.group(1)
            pk = os.path.join(self.log_dir, instance_name) + '.pickle'
            if os.path.exists(pk):
                with open(pk, "rb") as f:
                    [variable_features, constraint_features, edge_indices, edge_features, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con, solution] = pickle.load(f)    
            else:
                constraint_features, edge_indices, edge_features, variable_features, num_to_value, n, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con = get_a_new3(instance, self.random_feature)
                
                with open(solution_path, "rb") as f:
                    solution = pickle.load(f)[0]
                sol = []
                for i in range(n):
                    sol.append(solution[num_to_value[i]])
                    
                with open(pk, "wb") as f:
                    pickle.dump([variable_features, constraint_features, edge_indices, edge_features, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con, sol], f)
                solution = sol
             
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

def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output

def process(policy, data_loader, device, optimizer=None, tripartite=False):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            #print("QwQ")
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            if not tripartite:
                logits, select = policy(
                    batch.constraint_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.variable_features,
                )
            else :

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
            
            n = len(batch.variable_features)
            choose = {}
            for i in range(n):
                if(select[i] >= 0.5):
                    choose[i] = 0
                else:
                    choose[i] = 1
            new_idx_train = []
            for i in range(n):
                if(choose[i]):
                    new_idx_train.append(i)
            
            set_c = 0.7
            if(len(new_idx_train) < set_c * n):
                loss_select = (set_c - len(new_idx_train) / n) ** 2
            else:
                loss_select = 0
            #print(batch.constraint_features)
            #print(batch.edge_index)
            #print(batch.edge_attr)
            #print(batch.variable_features)
            # Index the results by the candidates, and split and pad them
            # logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            #loss = F.binary_cross_entropy(logits, batch.assignment)
            loss_func = torch.nn.MSELoss()
            #print(logits)
            #print(logits)
            #print(batch.assignment)
            #print(logits)
            loss = loss_func(logits[new_idx_train], batch.assignment[new_idx_train]) + loss_select
            
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            mean_loss += loss.item() * batch.num_graphs
            # mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    # mean_acc /= n_samples_processed
    return mean_loss

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
        
 #   tmp_coefficient = [abs(_) for _ in coefficient if _ != 0]
 #   tmp_coefficient = sorted(tmp_coefficient)
 #   threshold = tmp_coefficient[int(len(tmp_coefficient) * 0.5)]

    # for i in range(n):
    #     tmp = []
    #     coef = coefficient[i]
    #     if coef != 0:
    #         tmp.append(obj_type)
    #         tmp.append(1 if coef > 0 else -1)
    #         tmp.append(1 if abs(coef) > threshold else 0)
    #         if random_feature:
    #             tmp.append(random.random())
    #         obj_features.append(tmp)
    #         obj_num.append(i)
            
            
    # obj_features[0] = [cnt, obj_type]
    # if random_feature:
    #     obj_features[0].append(random.random())
    
    
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



def c(a):
    tmp = os.path.basename(a)
    tmp = re.match(r".*_([0-9]+)", tmp)
    tmp = tmp.group(1)
    return int(tmp) <= 9

def d(a):
    tmp = os.path.basename(a)
    tmp = re.match(r".*_([0-9]+)", tmp)
    tmp = tmp.group(1)
    return int(tmp) > 9 and int(tmp) < 16 

def train(
    train_data_dir: str,
    model_save_dir: Union[str, Path],
    log_dir: str,
    random_feature: bool = False,
    tripartite: bool = False,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    num_epochs: int = 20,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    This function trains a GNN policy on training data. 

    Args:
        data_path: Path to the data directory.
        model_save_path: Path to save the model.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        num_epochs: Number of epochs to train for.
        device: Device to use for training.
    """
    #训练路径
    train_data_path = train_data_dir
    # load samples from data_path and divide them
    DIR_BG = train_data_path + 'LP'
    DIR_SOL = train_data_path + 'Pickle'

    sample_names = os.listdir(DIR_BG)
    sample_files = [ (os.path.join(DIR_BG,name), os.path.join(DIR_SOL,name).replace('lp','pickle')) for name in sample_names if not c(name)]
#    sample_files = [ (os.path.join(DIR_BG,name), os.path.join(DIR_SOL,name).replace('lp','pickle')) for name in sample_names if d(name)]
    # TODO : modify !!!!

    train_files = sample_files[: int(0.9 * len(sample_files))]
    valid_files = sample_files[int(0.9 * len(sample_files)) :]

    train_data = GraphDataset(train_files, log_dir, random_feature, tripartite)
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle = False)
    valid_data = GraphDataset(valid_files, log_dir, random_feature, tripartite)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=batch_size, shuffle = False)

    policy = GNNPolicy(random_feature=random_feature, tripartite=tripartite).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loss = process(policy, train_loader, device, optimizer, tripartite)
        #valid_loss = process(policy, valid_loader, device, None)
        valid_loss = 0
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:0.3f}, Valid Loss: {valid_loss:0.3f}")
    
    model_save_path = f'{model_save_dir}/model_best.pkl'
    torch.save(policy.state_dict(), model_save_path)
    print(f"Trained parameters saved to {model_save_path}")

def parse_args():
    """
    This function parses the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, help="the train instances input folder")
    parser.add_argument("--model_save_dir", type=str, help="the model output directory")
    parser.add_argument("--log_dir", type=str, help="the train tmp file restore") 
    parser.add_argument("--random_feature", action='store_true', help="whether use random feature or not")
    parser.add_argument("--tripartite", action='store_true', help="whether use tripartite graph to encode problem")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to train for.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
