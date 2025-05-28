import torch
import os
import re
import gurobipy as gp
import numpy as np
import pyscipopt
import subprocess
import pickle
import random
import cplex
from .mod import Component, Graphencode2Predict, Predict2Modify, Cantsol, Cansol2M
from .help.NEURALDIVING.help import get_a_new2 as get_a_new2_gcn, get_a_new3 as get_a_new3_gcn, GraphDataset
from .help.LIGHT.help import get_a_new2 as get_a_new2_gat

from .help.LIGHT.EGAT_models import SpGAT
import torch_geometric


# different methods of predicting as different class
# gcn means graph convolutional network
# gurobi means using gurobi to find a feasible solution
# scip means using scip to find a feasible solution
# cplex means using cplex to find a feasible solution
# gat means graph attention network
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
        elif component == 'gat':
            cls = GAT
        else:
            raise ValueError("Predict component type is not defined")
        
        return super().__new__( cls )

    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)


    def work(self, input: Graphencode2Predict) -> Predict2Modify:...

class Gurobi(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
    
    # to find a feasible solution using gurobi
    # stop as soon as a feasible solution is found
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
        
        cansol = {}
        
        model = gp.read(self.instance)
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('SolutionLimit', 1)
        model.optimize()
        for var in model.getVars():
            cansol[var.VarName] = var.X

        self.end()
        return Cansol2M(model.ObjVal, cansol, model.MIPGap)
    
class SCIP(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
    
    # to find a feasible solution using SCIP
    # stop as soon as a feasible solution is found
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
    
        solver = pyscipopt.Model()
        solver.readProblem(self.instance)
        solver.setRealParam('limits/time', self.time_limit)
        solver.setIntParam("limits/solutions", 1)
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
    
    # to find a feasible solution using CPLEX
    # stop as soon as a feasible solution is found
    def work(self, input: Graphencode2Predict) -> Cansol2M:    
        
        self.begin()
        
        cansol = {}
        
        model = cplex.Cplex()
        model.read(self.instance)
        model.parameters.timelimit.set(self.time_limit)
        model.parameters.mip.limits.solutions.set(1)
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
        else :
            self.train_data_dir = None
 
    # to predict a solution using gcn (maybe infeasible)
    def work(self, input: Graphencode2Predict) -> Cantsol:    
        
        self.begin()
        # gcn network structure
        from .help.NEURALDIVING.graphcnn import GNNPolicy
        
        DEVICE = self.device     
        
        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*)_[0-9]+", instance_name)
        if instance_name == None:
            raise ValueError("instance name error!")
        else :
            instance_name = instance_name.group(1)
            
        
        # the model training needn't a path including sequence_name because now there is not hyper-parameters in graphencode layer or predict layer
        # so only contains instance_name and Graphencode and Predict components to ensure the reusability of the model
        model_dir = f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'
        model_path = f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/model_best.pkl'
        W = f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'

        if not os.path.isdir('../logs/'):
            os.mkdir('../logs')
        if not os.path.isdir(f'../logs/train/'):
            os.mkdir('../logs/train')
        if not os.path.isdir(f'../logs/train/{self.taskname}/'):
            os.mkdir(f'../logs/train/{self.taskname}')
        if not os.path.isdir(f'../logs/train/{self.taskname}/{instance_name}/'):
            os.mkdir(f'../logs/train/{self.taskname}/{instance_name}')
        if not os.path.isdir(f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
            os.mkdir(f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
        if not os.path.isdir(f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'):
            os.mkdir(f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')
        
            
        # check the model, if there is not then train using train instances
        if not os.path.exists(model_path):
            if not os.path.isdir('../Model/'):
                os.mkdir('../Model/')
            if not os.path.isdir(f'../Model/{self.taskname}'):
                os.mkdir(f'../Model/{self.taskname}')
            if not os.path.isdir(f'../Model/{self.taskname}/{instance_name}'):
                os.mkdir(f'../Model/{self.taskname}/{instance_name}')
            if not os.path.isdir(f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
                os.mkdir(f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
            if not os.path.isdir(f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}'):
                os.mkdir(f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')

            # train the model using the train program
            exec = ["python", "./help/NEURALDIVING/train.py", "--train_data_dir", f"{self.train_data_dir}",
                   "--model_save_dir", f"{model_dir}", "--log_dir", f"{W}", "--device", f"{self.device}"]
            # check if using tripartite and random feature
            if self.sequence_name[0][-1] == 'r':
                exec.append("--random_feature")
            if self.sequence_name[0][0] == 't':
                exec.append("--tripartite")
            subprocess.run(exec)
        
        tripartite = True if self.sequence_name[0][0] == 't' else False
            
        policy = GNNPolicy(random_feature=True if self.sequence_name[0][-1] == 'r' else False, tripartite=tripartite).to(DEVICE)
        policy.load_state_dict(torch.load(model_path, policy.state_dict()))
        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*_[0-9]+)\.lp", instance_name)
        instance_name = instance_name.group(1)
        
        pk = os.path.join(W, instance_name) + '.pickle'
    
        # if the pickle file(containing the features) does not exist, then generate the features and save them
        if not os.path.exists(pk):
            if not tripartite:
                constraint_features, edge_indices, edge_features, variable_features, num_to_value, n = get_a_new2_gcn(self.instance, random_feature=True if self.sequence_name[0][-1] == 'r' else False)            
                sol = []
                with open(pk, "wb") as f:
                    pickle.dump([variable_features, constraint_features, edge_indices, edge_features, sol], f)
                    
            else :
                constraint_features, edge_indices, edge_features, variable_features, num_to_value, n, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con = get_a_new3_gcn(self.instance, random_feature=True if self.sequence_name[0][-1] == 'r' else False)
                sol = []
                with open(pk, "wb") as f:
                    pickle.dump([variable_features, constraint_features, edge_indices, edge_features, obj_features, obj_variable_val, obj_constraint_val, edge_obj_var, edge_obj_con, sol], f)
                
        # load the data using Dataset batch with a batch whose size is 1 
        file = [pk]
        data = GraphDataset(file, tripartite=tripartite)
        loader = torch_geometric.loader.DataLoader(data, batch_size = 1)

        logits, select = None, None
        for batch in loader:
            batch = batch.to(self.device)
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

        self.end()
        return Cantsol(logits, select)

class GAT(Predict):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        self.time_limit = kwargs.get("time_limit") or 10
        if "train_data_dir" in kwargs:
            self.train_data_dir = kwargs["train_data_dir"]
        else :
            self.train_data_dir = None
 
    # to predict a solution using gat (maybe infeasible)
    def work(self, input: Graphencode2Predict) -> Cantsol:    
        
        self.begin()
        
        device = self.device
        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*)_[0-9]+", instance_name)
        if instance_name == None:
            raise ValueError("instance name error!")
        else :
            instance_name = instance_name.group(1)
            
        # the model training needn't a path including sequence_name because now there is not hyper-parameters in graphencode layer or predict layer
        # so only contains instance_name and Graphencode and Predict components to ensure the reusability of the model
        model_dir = f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'
        model_path = f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/model_best.pkl'
        W = f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'

        if not os.path.isdir('../logs/'):
            os.mkdir('../logs')
        if not os.path.isdir(f'../logs/train/'):
            os.mkdir('../logs/train')
        if not os.path.isdir(f'../logs/train/{self.taskname}/'):
            os.mkdir(f'../logs/train/{self.taskname}')
        if not os.path.isdir(f'../logs/train/{self.taskname}/{instance_name}/'):
            os.mkdir(f'../logs/train/{self.taskname}/{instance_name}')
        if not os.path.isdir(f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
            os.mkdir(f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
        if not os.path.isdir(f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}/'):
            os.mkdir(f'../logs/train/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')

        # check the model, if there is not then train using train instances
        if not os.path.exists(model_path):
            if not os.path.isdir('../Model/'):
                os.mkdir('../Model/')
            if not os.path.isdir(f'../Model/{self.taskname}'):
                os.mkdir(f'../Model/{self.taskname}')
            if not os.path.isdir(f'../Model/{self.taskname}/{instance_name}'):
                os.mkdir(f'../Model/{self.taskname}/{instance_name}')
            if not os.path.isdir(f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}'):
                os.mkdir(f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}')
            if not os.path.isdir(f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}'):
                os.mkdir(f'../Model/{self.taskname}/{instance_name}/{self.sequence_name[0]}/{self.sequence_name[1]}')

            # train the model using the train program
            exec = ["python", "./help/LIGHT/train.py", "--train_data_dir", f"{self.train_data_dir}",
                    "--model_save_dir", f"{model_dir}", "--log_dir", f"{W}", "--device", f"{self.device}", "--lr", "1e-4", "--alpha", "2e-4", "--no-cuda"]
      
            # check if using random feature
            if self.sequence_name[0][-1] == 'r':
                exec.append("--random_feature")
            subprocess.run(exec)
                
        instance_name = os.path.basename(self.instance)
        instance_name = re.match(r"(.*_[0-9]+)\.lp", instance_name)
        instance_name = instance_name.group(1)
        
        pk_feature = os.path.join(W, instance_name) + '.pickle'
        if not os.path.exists(pk_feature):
            constraint_features, edge_indices, edge_features, variable_features, num_to_value, n = get_a_new2_gat(self.instance, random_feature=True if self.sequence_name[0][-1] == 'r' else False)
            sol = []
            with open(pk_feature, "wb") as f:
                pickle.dump([variable_features, constraint_features, edge_indices, edge_features, sol], f)

        with open(pk_feature, 'rb') as f:
            problem = pickle.load(f)
                
        variable_features = problem[0]
        constraint_features = problem[1]
        edge_indices = problem[2]
        edge_feature = problem[3]
        n = len(variable_features)
        var_size = len(variable_features[0])
        m = len(constraint_features)
        con_size = len(constraint_features[0])
        edge_num = len(edge_indices[0])

        edgeA = []
        edgeB = []
        edge_features = []
        for i in range(edge_num):
            edge_feature[i][0] /= n
        for i in range(edge_num):
            edgeA.append([edge_indices[1][i], edge_indices[0][i] + n])
            edgeB.append([edge_indices[0][i] + n, edge_indices[1][i]])
            edge_features.append(edge_feature[i])
        edgeA = torch.as_tensor(edgeA)
        edgeB = torch.as_tensor(edgeB)
        edge_features = torch.as_tensor(edge_features)

        for i in range(m):
            for j in range(var_size - con_size):
                constraint_features[i].append(0)


        features = variable_features + constraint_features
        features = torch.as_tensor(features)

        ##Predict
        #FENNEL
        partition_num = int(n / 20000) if n > 20000 else 1
        partition_var = []
        for i in range(partition_num):
            partition_var.append([])
        vertex_num = n + m
        edge_num = 0

        edge = []
        edge_val = []
        for i in range(vertex_num):
            edge.append([])
            edge_val.append([])
        for i in range(len(edgeA)):
            edge[edgeA[i][0]].append(edgeA[i][1])
            edge_val[edgeA[i][0]].append(1)
            edge[edgeA[i][1]].append(edgeA[i][0])
            edge_val[edgeA[i][1]].append(1)
            edge_num += 2

        alpha = (partition_num ** 0.5) * edge_num / (vertex_num ** (2 / 3))
        gamma = 1.5
        balance = 1.1

        visit = np.zeros(vertex_num, int)
        order = []
        for i in range(vertex_num):
            if(visit[i] == 0):
                q = []
                q.append(i)
                now = 0
                while(now < len(q)):
                    order.append(q[now])
                    for neighbor in edge[q[now]]:
                        if(visit[neighbor] == 0):
                            q.append(neighbor)
                            visit[neighbor] = 1
                    now += 1

        color = np.zeros(vertex_num, int)
        for i in range(vertex_num):
            color[i] = -1
        cluster_num = np.zeros(partition_num)
        score = np.zeros(partition_num, float)
        for i in range(vertex_num):
            now_vertex = order[i]
            load_limit = balance * vertex_num / partition_num
            for j in range(len(edge[now_vertex])):
                neighbor = edge[now_vertex][j]
                if(color[neighbor] != -1):
                    score[color[neighbor]] += edge_val[now_vertex][j]
            
            now_score = -2e9
            now_site = -1
            for j in range(len(edge[now_vertex])):
                neighbor = edge[now_vertex][j]
                if(color[neighbor] != -1):
                    if(score[color[neighbor]] > now_score):
                        now_score = score[color[neighbor]]
                        now_site = color[neighbor]
            neighbor = random.randint(0, partition_num - 1)
            if(score[neighbor] > now_score):
                now_score = score[neighbor]
                now_site = neighbor
            
            color[now_vertex] = now_site
            score[now_site] += alpha * gamma * (cluster_num[now_site] ** (gamma - 1))
            cluster_num[now_site] += 1
            score[now_site] -= alpha * gamma * (cluster_num[now_site] ** (gamma - 1))
            if(now_vertex < n):
                partition_var[now_site].append(now_vertex - n)

        color_site_to_num = []
        num_to_color_site = []
        color_site_num = []
        color_edgeA = []
        color_edgeB = []
        color_edge_features = []
        color_features = []
        color_edge_to_num = []
        for i in range(partition_num):
            color_site_to_num.append([])
            color_site_num.append(0)
            color_features.append([])
            color_edgeA.append([])
            color_edgeB.append([])
            color_edge_features.append([])
            color_edge_to_num.append([])

        for i in range(vertex_num):
            num_to_color_site.append(color_site_num[color[i]])
            color_site_num[color[i]] += 1
            color_site_to_num[color[i]].append(i)
            color_features[color[i]].append(features[i])

        edge_num = len(edge_indices[0])
        for i in range(edge_num):
            if(color[edge_indices[1][i]] == color[edge_indices[0][i] + n]):
                now_color = color[edge_indices[1][i]]
                color_edgeA[now_color].append([num_to_color_site[edge_indices[1][i]], num_to_color_site[edge_indices[0][i] + n]])
                color_edgeB[now_color].append([num_to_color_site[edge_indices[0][i] + n], num_to_color_site[edge_indices[1][i]]])
                color_edge_features[now_color].append(edge_feature[i])
                color_edge_to_num[now_color].append(i)

        path_model = model_path
        model = SpGAT(nfeat=features.shape[1],    # Feature dimension
                    nhid=64,                    # Feature dimension of each hidden layer
                    nclass=1,                   # Number of classes
                    dropout=0.5,                # Dropout
                    nheads=6,                   # Number of heads
                    alpha=0.2)                  # LeakyReLU alpha coefficient

        state_dict_load = torch.load(path_model)
        model.load_state_dict(state_dict_load)
        model.to(self.device)

        def compute_test(features, edgeA, edgeB, edge_features):
            model.eval()
            output, select, new_edge_feat = model(features, edgeA, edgeB, edge_features)
            return(output, select, new_edge_feat)

        predict = [0] * (n + m)
        select = [0] * (n + m)
        new_edge_feat = [0] * edge_num
        for i in range(partition_num):
            now_predict, now_select, now_new_edge_feat = compute_test(torch.tensor(np.array([item.cpu().detach().numpy() for item in color_features[i]])).cuda().float().to(device), torch.as_tensor(color_edgeA[i]).to(device), torch.as_tensor(color_edgeB[i]).to(device), torch.as_tensor(color_edge_features[i]).float().to(device))
            for j in range(len(color_site_to_num[i])):
                if(color_site_to_num[i][j] < n):
                    predict[color_site_to_num[i][j]] = now_predict[j].cpu().detach().numpy()
                    select[color_site_to_num[i][j]] = now_select[j].cpu().detach().numpy()
            for j in range(len(color_edge_to_num[i])):
                new_edge_feat[color_edge_to_num[i][j]] = now_new_edge_feat[j].cpu().detach().numpy()

        self.end()
        
        return Cantsol(predict, select)


