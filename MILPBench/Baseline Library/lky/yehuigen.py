import torch
import os
import re
import gurobipy as gp
import numpy as np
import pyscipopt
import subprocess
import pickle
import random
import json
import cplex
import argparse
import torch.nn as nn
import torch.nn.functional as F
import time
from gurobipy import *
from typing import Type, cast
import torch_geometric

parser = argparse.ArgumentParser(description="receive select instruction from higher level")
parser.add_argument("--device", required=True, choices=["cpu", "cuda", "cuda:2", "cuda:1", "cuda:3"], help="cpu or cuda")
parser.add_argument("--taskname", required=True, choices=["MVC", "IS", "MIKS", "SC", "MIKSC"], help="taskname")
parser.add_argument("--instance_path", type=str, required=True, help="the task instance path")
parser.add_argument("--whole_time_limit", type=int, help="time limit for whole process")
parser.add_argument("--model_path", type=str, help="model path")


parser.add_argument("--search_ACP_LNS_block", type=int, help="ACP / LNS block parameter")
parser.add_argument("--search_ACP_LNS_max_turn_ratio", type=float, help="ACP / LNS max_turn_ratio parameter")
args = parser.parse_args()

device = args.device
instance = args.instance_path
taskname = args.taskname
model_path = args.model_path


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    def __init__(self, node_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.node_features = node_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(node_features, out_features)))  
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features + 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, node, edge, edge_feature):
        dv = device if node.is_cuda else 'cpu'
        #dv = 'cpu'
        N = node.size()[0]
        edge = edge.t()
        edge = torch.nan_to_num(edge, nan=0.0, posinf=1e10, neginf=-1e10)
        assert not torch.isnan(edge).any()
        #print(

        h = torch.mm(node, self.W)
#        print("SSSSSSSSSSSSSS", node.shape, node)
#        print("WWWWWWWWWWWWWWW", self.W.shape, self.W)
#        print("!!!!!!!!!!!!!!!!", h.shape, h)
#        we = torch.isnan(h)
#        if we.any():
#            print(torch.nonzero(we))
#            print(h[we])
#        print("_______________________________________________")

        # TODO
        # sum = 0
        # for i in range(node.shape[1]):
        #     if i == 257:
        #         print(node[1][i], self.W[i][0])
        #     sum = sum + node[1][i] * self.W[i][0]
        #     print(i, sum)
            
        # h: N x out
        h = torch.nan_to_num(h, nan=0.0, posinf=1e10, neginf=-1e10)
        
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        #print(torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1))
        #print(edge_feature)
        edge_h = torch.cat((torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1), edge_feature), dim = 1).t()
        edge_h = torch.nan_to_num(edge_h, nan=0.0, posinf=1e10, neginf=-1e10)
        assert not torch.isnan(edge_h).any()
        # edge: (2*D + 1) x E

        # TODO
        # tmp = edge_h
        # if torch.isinf(tmp).any() or torch.isnan(tmp).any():
        #     print("edge_hLKY")
        #     wee = torch.isnan(tmp)
        #     print(torch.nonzero(wee))
        #     print("LKY")
        #     wee = torch.isinf(tmp)
        #     print(torch.nonzero(wee))

        # tmp = self.a.mm(edge_h).squeeze()
        # tmp = -self.leakyrelu(tmp)
        # print("TMPTMPTMPT")
        # print(tmp.max().item())
        # print(tmp.min().item())
        # if torch.isinf(tmp).any() or torch.isnan(tmp).any():
        #     print("tmpLKY")
        #     wee = torch.isnan(tmp)
        #     print(torch.nonzero(wee))
        #     print("LKY")
        #     wee = torch.isinf(tmp)
        #     print(torch.nonzero(wee))

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        tmp = edge_e

        edge_e = torch.nan_to_num(edge_e, nan=0.0, posinf=1e10, neginf=-1e10)
        assert not torch.isnan(edge_e).any()
        # attention, edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1   

        #edge_e = self.dropout(edge_e)
        # edge_e: E
        
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        h_prime = torch.nan_to_num(h_prime, nan=0.0, posinf=1e10, neginf=-1e10)
        
        #
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out


        h_prime = h_prime.div(e_rowsum)
        h_prime = torch.where(torch.isnan(h_prime), torch.full_like(h_prime, 0), h_prime)
        h_prime = torch.add(h, h_prime)
        # h_prime: N x out
        h_prime = torch.nan_to_num(h_prime, nan=0.0, posinf=1e10, neginf=-1e10)
        
        assert not torch.isnan(h_prime).any()

        #print(h.size(), h_prime.size())

        if self.concat:
            # if this layer is not last layer,
            return [F.elu(h_prime), edge_e.reshape(edge_e.size()[0], 1)]
        else:
            # if this layer is last layer,
            return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.node_features) + ' -> ' + str(self.out_features) + ')'



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        '''
        Function Description:
        Initializes the model by defining the size of the feature space, and sets up layers for encoding decision variables, edge features, and constraint features. 
        It includes two semi-convolutional attention layers and a final output layer.
        - nfeat: Initial feature dimension.
        - nhid: Dimension of the hidden layers.
        - nclass: Number of classes; for 0-1 integer programming, this would be 2.
        - dropout: Dropout rate.
        - alpha: Coefficient for leakyReLU.
        - nheads: Number of heads in the multi-head attention mechanism.
        Hint: Use the pre-written SpGraphAttentionLayer for the attention layers.
        '''
        super(SpGAT, self).__init__()
        self.dropout = dropout
        embed_size = 64
        self.input_module = torch.nn.Sequential(
            torch.nn.Linear(nfeat, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
        )
        self.attentions_u_to_v = [SpGraphAttentionLayer(embed_size,
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_u_to_v):
            self.add_module('attention_u_to_v_{}'.format(i), attention)
        self.attentions_v_to_u = [SpGraphAttentionLayer(embed_size,
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_v_to_u):
            self.add_module('attention_v_to_u_{}'.format(i), attention)

        self.out_att_u_to_v = SpGraphAttentionLayer(nhid * nheads, 
                                               embed_size, 
                                               dropout=dropout, 
                                               alpha=alpha, 
                                               concat=False)
        self.out_att_v_to_u = SpGraphAttentionLayer(nhid * nheads, 
                                               embed_size, 
                                               dropout=dropout, 
                                               alpha=alpha, 
                                               concat=False)
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(embed_size, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, embed_size),
            #torch.nn.LogSoftmax(dim = 0),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_size, nclass, bias=False),
            #torch.nn.Sigmoid()
        )
        
        # self.select_module = torch.nn.Sequential(
        #     torch.nn.Linear(embed_size, embed_size),
        # #torch.nn.LogSoftmax(dim = 0),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(embed_size, embed_size),
        #     #torch.nn.LogSoftmax(dim = 0),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(embed_size, nclass, bias=False),
        #     #torch.nn.Sigmoid()
        # )
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, edgeA, edgeB, edge_feat):
        '''
        Function Description:
        Executes the forward pass using the provided constraint, edge, and variable features, processing them through an EGAT to produce the output.

        Parameters:
        - x: Features of the variable and constraint nodes.
        - edgeA, edgeB: Information about the edges.
        - edge_feat: Features associated with the edges.

        Return: The result after the forward propagation.
        '''
        #print(x)
        x = self.input_module(x)
            
        #x = F.dropout(x, self.dropout, training=self.training)
        #print(x)
        new_edge = torch.cat([att(x, edgeA, edge_feat)[1] for att in self.attentions_u_to_v], dim=1)

        # TODO
        # cnt = 0
        # for att in self.attentions_u_to_v:
        #     cnt += 1 
        #     print("CNT: ", cnt)
        #     if cnt != 5:
        #         continue
        #     tmp = att(x, edgeA, edge_feat)[0]
        #     if torch.isinf(tmp).any() or torch.isnan(tmp).any():
        #         print(cnt)
        #         print("NANANANANANANANA")
        #         wee = torch.isnan(tmp)
        #         print(torch.nonzero(wee))
        #         print("INFINFINFINFINFINFINFINFIN")
        #         wee = torch.isinf(tmp)
        #         print(torch.nonzero(wee))

        x = torch.cat([att(x, edgeA, edge_feat)[0] for att in self.attentions_u_to_v], dim=1)
#        if torch.isinf(x).any():
#            wee = torch.isinf(x)
#            print(torch.nonzero(wee))
#        print("INFINFINFINFINFINFINFINFINFINFINFINFINFINFINFIN")
        x = self.out_att_u_to_v(x, edgeA, edge_feat)
        new_edge = torch.mean(new_edge, dim = 1).reshape(new_edge.size()[0], 1)
        #x = self.softmax(x)
        new_edge_ = torch.cat([att(x, edgeB, new_edge)[1] for att in self.attentions_v_to_u], dim=1)
        x = torch.cat([att(x, edgeB, new_edge)[0] for att in self.attentions_v_to_u], dim=1)
        x = self.out_att_v_to_u(x, edgeB, new_edge)
        new_edge_ = torch.mean(new_edge_, dim = 1).reshape(new_edge_.size()[0], 1)

#        y = self.select_module(x)
        x = self.output_module(x)
        x = self.softmax(x)

#        return x.squeeze(-1), y.squeeze(-1), new_edge_
#        return x.squeeze(-1), new_edge_
        print(x.shape)
        return x, new_edge_


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
        now_variable_features.append(0)
        now_variable_features.append(1)        
        # if(lower_bound[i] == float("-inf")):
        #     now_variable_features.append(0)
        #     now_variable_features.append(0)
        # else:
        #     now_variable_features.append(1)
        #     now_variable_features.append(lower_bound[i])
        # if(upper_bound[i] == float("inf")):
        #     now_variable_features.append(0)
        #     now_variable_features.append(0)
        # else:
        #     now_variable_features.append(1)
        #     now_variable_features.append(upper_bound[i])
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
#        now_constraint_features.append(constraint_type[i])
        if constraint_type[i] == 1:
            now_constraint_features.append(1)
            now_constraint_features.append(0)
            now_constraint_features.append(0)
        elif constraint_type[i] == 2:
            now_constraint_features.append(0)
            now_constraint_features.append(1)
            now_constraint_features.append(0)
        elif constraint_type[i] == 3:
            now_constraint_features.append(0)
            now_constraint_features.append(0)
            now_constraint_features.append(1)
        if random_feature:
            now_constraint_features.append(random.random())
        constraint_features.append(now_constraint_features)
    
    for i in range(m):
        for j in range(k[i]):
            edge_indices[0].append(i)
            edge_indices[1].append(site[i][j])
            edge_features.append([value[i][j]])

    return constraint_features, edge_indices, edge_features, variable_features, num_to_value, n


def get_a_new22(instance, random_feature = False):
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

    return constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value


def predict():
    print("GAT predict...")
    instance_name = os.path.basename(instance)
    instance_name = re.match(r"(.*)_[0-9]+", instance_name)
    if instance_name == None:
        raise ValueError("instance name error!")
    else :
        instance_name = instance_name.group(1)
        
    constraint_features, edge_indices, edge_features, variable_features, num_to_value, n = get_a_new2(instance, random_feature=True)
            
    edge_feature = edge_features
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

    idx_test = torch.tensor(range(n))

    ##Predict
    #FENNEL
    partition_num = int(n / 20000)
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
    #                    nclass=1,                   # Number of classes
                nclass=2,                   # Number of classes 
                dropout=0.5,                # Dropout
                nheads=6,                   # Number of heads
                alpha=0.2)                  # LeakyReLU alpha coefficient
    state_dict_load = torch.load(path_model)
    model.load_state_dict(state_dict_load)
    model.to(device)

    def compute_test(features, edgeA, edgeB, edge_features):
        model.eval()
    #            output, select, new_edge_feat = model(features, edgeA, edgeB, edge_features)
        output, new_edge_feat = model(features, edgeA, edgeB, edge_features)
        #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        #acc_test = accuracy(output[idx_test], labels[idx_test])
        #print("Test set results:",
        #      "loss= {:.4f}".format(loss_test.data.item()))
    #            return(output, select, new_edge_feat)
        return (output, new_edge_feat)


    predict = [0] * (n + m)
    select = [0] * (n + m)
    new_edge_feat = [0] * edge_num
    for i in range(partition_num):
    #            now_predict, now_select, now_new_edge_feat = compute_test(torch.tensor(np.array([item.cpu().detach().numpy() for item in color_features[i]])).cuda().float().to(device), torch.as_tensor(color_edgeA[i]).to(device), torch.as_tensor(color_edgeB[i]).to(device), torch.as_tensor(color_edge_features[i]).float().to(device))
        now_predict, now_new_edge_feat = compute_test(torch.tensor(np.array([item.cpu().detach().numpy() for item in color_features[i]])).cuda().float().to(device), torch.as_tensor(color_edgeA[i]).to(device), torch.as_tensor(color_edgeB[i]).to(device), torch.as_tensor(color_edge_features[i]).float().to(device))

        for j in range(len(color_site_to_num[i])):
            if(color_site_to_num[i][j] < n):
                tmp = now_predict[j].cpu().detach().numpy()
                predict[color_site_to_num[i][j]] = 1 if tmp[1] > 0.5 else 0
                select[color_site_to_num[i][j]] = tmp[predict[color_site_to_num[i][j]]]
    #                    select[color_site_to_num[i][j]] = now_select[j].cpu().detach().numpy()
        for j in range(len(color_edge_to_num[i])):
            new_edge_feat[color_edge_to_num[i][j]] = now_new_edge_feat[j].cpu().detach().numpy()

    return predict, select

def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, now_col, constr_flag, lower_bound, upper_bound, value_type):
    '''
    Function Explanation:
    This function solves a problem instance using the SCIP solver based on the provided parameters.

    Parameter Explanation:
    - n: The number of decision variables in the problem instance.
    - m: The number of constraints in the problem instance.
    - k: k[i] indicates the number of decision variables in the i-th constraint.
    - site: site[i][j] indicates which decision variable the j-th decision variable in the i-th constraint is.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] indicates the type of the i-th constraint, where 1 represents <= and 2 represents >=.
    - coefficient: coefficient[i] indicates the coefficient of the i-th decision variable in the objective function.
    - time_limit: The maximum solving time.
    - obj_type: Specifies whether the problem is a maximization or minimization problem.
    - now_sol: The current solution.
    - now_col: Dimensionality reduction flags for decision variables.
    - constr_flag: Dimensionality reduction flags for constraints.
    - lower_bound: Lower bounds for decision variables.
    - upper_bound: Upper bounds for decision variables.
    - value_type: The type of decision variables (e.g., integer or continuous variables).
    '''
    # Get the start time
    begin_time = time.time()

    # Define the solver model
    model = Model("Gurobi")
    model.feasRelaxS(0,False,False,True)

    # Set up variable mappings
    site_to_new = {}
    new_to_site = {}
    new_num = 0

    # Define new_num decision variables x[]
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

    # Set the objective function and optimization goal (maximize/minimize)
    coeff = 0
    for i in range(n):
        if(now_col[i] == 1):
            coeff += x[site_to_new[i]] * coefficient[i]
        else:
            coeff += now_sol[i] * coefficient[i]
    if(obj_type == 'maximize'):
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)
    
    # Add m constraints
    for i in range(m):
        if(constr_flag[i] == 0):
            continue
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
            else:
                model.addConstr(constr >= constraint[i])
        else:
            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    #print(now_col)
            else:
                if(constr < constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    #print(now_col)
    
    # Set the maximum solving time
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    
    # Optimize the solution
    model.optimize()
    #print(time.time() - begin_time)
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
            
        return new_sol, model.ObjVal
    except:
        return -1, -1

def repair(logits, select, time_limit):
    print("Nr repair...")
    constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value=get_a_new2(instance)

    color = np.zeros(n) # 1: discard
    
    if type(logits) == list:
        now_sol = np.array(logits)
    else:
        now_sol = logits.to('cpu').detach().numpy() 

    for i in range(n):
        if(value_type[i] != 'C'):
            now_sol[i] = int(now_sol[i] + 0.5)
        now_sol[i] = min(now_sol[i], upper_bound[i])
        now_sol[i] = max(now_sol[i], lower_bound[i])


    F = 0
    result_pair = (0, 0, 0)
    for LL in range(3): # 3 times repair. If can't then fail
        discard = []
        for j in range(m):
            constr = 0
            flag = 0
            for l in range(k[j]):
                if color[site[j][l]] == 1:
                    flag = 1
                    break
                else:
                    constr += now_sol[site[j][l]] * value[j][l]
            if flag == 1:
                for l in range(k[j]):
                    if color[site[j][l]] == 0:
                        discard.append(site[j][l])
                continue
            if(constraint_type[j] == 1):
                if(constr > constraint[j]):
                    for l in range(k[j]):
                        if color[site[j][l]] == 0:
                            discard.append(site[j][l])
            else:
                if(constr < constraint[j]):
                    for l in range(k[j]):
                        if color[site[j][l]] == 0:
                            discard.append(site[j][l])
        for i in discard:
            color[i] = 1
            
        flag, sol, obj, gap = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, color, lower_bound, upper_bound, value_type)

        if flag == 1:
            result_pair = (sol, obj, gap)
            F = 1
            break
    
    
    cansol = {}
    for i in range(n):
        cansol[num_to_value[i]] = result_pair[0][i]

    return result_pair[1], cansol, result_pair[2]


def split_problem(lp_file):
    '''
    Pass in an LP file and split the given problem
    '''
    model = gp.read(lp_file)
    value_to_num = {}
    value_num = 0
    #n represents the num of decision variables
    #m represents the num of constrains
    #k[i] represents the number of decision variables in the i-th constraint
    #site[i][j] represents which decision variable is the j-th decision variable in the i-th constraint
    #value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint
    #constraint[i] represents the number on the right-hand side of the i-th constraint
    #constraint_type[i] represents the type of the i-th constraint, where 1 indicates '<', 2 indicates '>', and 3 indicates '='
    #coefficient[i] represents the coefficient of the i-th decision variable in the objective function
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
            value_num += 1
        coefficient[value_to_num[val.VarName]] = val.Obj
        lower_bound[value_to_num[val.VarName]] = val.LB
        upper_bound[value_to_num[val.VarName]] = val.UB
        value_type[value_to_num[val.VarName]] = val.Vtype

    #1 for maximize，-1 for minimize
    obj_type = model.ModelSense
    if(obj_type == -1):
        obj_type = 'maximize'
    else:
        obj_type = 'minimize'
    return n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num


def search(objval, cansol, gap, time_limit, block, max_turn_ratio, result_list):
    print("ACP search...")
    #Set KK as the initial number of blocks and PP as the selected number of blocks to optimize after dividing the constraints into KK blocks
    KK = block
    PP = 1
    max_turn = 5
    epsilon = 0.01
    #Retrieve the problem model after splitting and create a new folder named "ACP_Pickle"
    max_turn_ratio = max_turn_ratio
    time_limit = time_limit
    max_turn_time = max_turn_ratio * time_limit
    n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num = split_problem(instance)    
    # Build a variable dictionar
    num_to_value = {value : key for key, value in value_to_num.items()}

    #Get the start time
    begin_time = time.time()
    
    #Initialize the initial solution and initial answer, where the initial solution is the worst initial solution
    
    ans, ansx = input.objval, []
    
    tmp = gp.read(instance)
    for var in tmp.getVars():
        ansx.append(input.cansol[var.VarName])
        
    print(f"初始解目标值为：{ans}")
    
    #Constraint block labels, where cons_color[i] represents which block the i-th constraint belongs to
    cons_color = np.zeros(m, int)
    
    #Initialize the initial solution and initial answer, where the initial solution is the worst initial solution
    last = ans
    
    #Initialize the number of rounds below the threshold, which can be either 0 or 1 with little difference.
    turn = 1
    #Continue to the next iteration as long as it hasn't reached the maximum running time

    now_sol, now_time = [], []

    while(time.time() - begin_time <= time_limit):
        print("KK = ", KK)
        print("PP = ", PP)
        #Randomly divide the decision variables into KK blocks
        for i in range(m):
            cons_color[i] = random.randint(1, KK)
        now_cons_color = 1
        #Set all decision variables involved in a randomly selected constraint block as variables to be optimized
        #color[i] = 1 indicates that the i-th decision variable is selected as a variable to be optimized
        color = np.zeros(n, int)
        now_color = 1
        color_num = 0
        for i in range(m):
            if(PP == 1 and cons_color[i] == now_cons_color):
                for j in range(k[i]):
                    color[site[i][j]] = 1
                    color_num += 1
            if(PP > 1 and cons_color[i] != now_cons_color):
                for j in range(k[i]):
                    color[site[i][j]] = 1
                    color_num += 1
        #site_to_color[i]represents which decision variable is the i-th decision variable in this block
        #color_to_site[i]represents which decision variable is mapped to the i-th decision variable in this block
        #vertex_color_num represents the number of decision variables in this block currently
        site_to_color = np.zeros(n, int)
        color_to_site = np.zeros(n, int)
        vertex_color_num = 0
        #Define the model to solve
        model = gp.Model("ACP")
        x = []
        for i in range(n):
            if(color[i] == now_color):
                color_to_site[vertex_color_num] = i
                site_to_color[i] = vertex_color_num
                vertex_color_num += 1
                if(value_type[i] == 'B'):
                    now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY)
                elif(value_type[i] == 'I'):
                    now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER)
                else:
                    now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS)
                x.append(now_val)
        #Define decision variables x[]
        #x = model.addMVar((vertex_color_num), lb = 0, ub = 1, vtype = GRB.BINARY)  #lb is the lower bound for the variable， ub is the upper bound for the variables
        #Set up the objective function and optimization objective (maximization/minimization), only optimizing the variables selected for optimization
        objsum = 0
        objtemp = 0
        for i in range(n):
            if(color[i] == now_color):
                objsum += x[site_to_color[i]] * coefficient[i]
            else:
                objtemp += ansx[i] * coefficient[i]
        if(obj_type == 'maximize'):
            model.setObjective(objsum, GRB.MAXIMIZE)
        else:
            model.setObjective(objsum, GRB.MINIMIZE)
        #Add m constraints, only adding those constraints that involve variables in this block
        for i in range(m): 
            flag = 0
            constr = 0
            for j in range(k[i]):
                if(color[site[i][j]] == now_color):
                    flag = 1
                    constr += x[site_to_color[site[i][j]]] * value[i][j]
                else:
                    constr += ansx[site[i][j]] * value[i][j]
            if(flag):
                if(constraint_type[i] == 1):
                    model.addConstr(constr <= constraint[i])
                elif(constraint_type[i] == 2):
                    model.addConstr(constr >= constraint[i])
                else:
                    model.addConstr(constr == constraint[i])
        
        #Set the maximum solving time
        model.setParam('TimeLimit', min(max(time_limit - (time.time() - begin_time), 0), max_turn_time))
        #Optimize
        model.optimize()
        
        try:
            #Calculate the current objective value
            temp = model.ObjVal + objtemp
            print(f"当前目标值为：{temp}")
            bestX = []
            for i in range(vertex_color_num):
                bestX.append(x[i].X)
            #print(bestX)

            if(obj_type == 'maximize'):
                #Update the current best solution and best ans
                if(temp > ans):
                    for i in range(vertex_color_num):
                        ansx[color_to_site[i]] = bestX[i]
                    ans = temp
                #Adaptive block number change
                if((ans - last <= epsilon * ans)):
                    turn += 1
                    if(turn == max_turn):
                        if(KK > 2 and PP == 1):
                            KK -= 1
                        else:
                            KK += 1
                            PP += 1
                        turn = 1
                else:
                    turn = 0
            else:
                if(temp < ans):
                    for i in range(vertex_color_num):
                        ansx[color_to_site[i]] = bestX[i]
                    ans = temp
                #Adaptive block number change
                if((last - ans <= epsilon * ans)):
                    turn += 1
                    if(turn == max_turn):
                        if(KK > 2 and PP == 1):
                            KK -= 1
                        else:
                            KK += 1
                            PP += 1
                        turn = 1
                else:
                    turn = 0
                
            now_sol.append(ans)
            now_time.append(time.time() - begin_time)
            print(now_sol[-1], now_time[-1])
                                
            if(model.MIPGap >= 0.0001):
#                if(model.MIPGap != 0):
                if(KK == 2 and PP > 1):
                    KK -= 1
                    PP -= 1
                else:
                    KK += 1
                turn = 0
            last = ans
        except:
            try:
                model.computeIIS()
                if(KK > 2 and PP == 1):
                    KK -= 1
                else:
                    KK += 1
                    PP += 1
                turn = 1
            except:
                if(KK == 2 and PP > 1):
                    KK -= 1
                    PP -= 1
                else:
                    KK += 1
                turn = 0                    
            print("This turn can't improve more")
    new_ansx = {}
    for i in range(len(ansx)):
        new_ansx[num_to_value[i]] = ansx[i]

    for _ in range(len(now_sol)):
        result_list.append((result_list[0][0] + now_time[_], now_sol[_]))


result_list_obj_time = []
start_time = time.time()
whole_time_limit = args.whole_time_limit
    
predict_, select = predict()
objval, cansol, gap = repair(predict_, select, whole_time_limit - (time.time() - start_time))
result_list_obj_time.append((time.time() - start_time, objval))    
search(objval, cansol, gap, whole_time_limit - (time.time() - start_time), args.search_ACP_LNS_block, args.search_ACP_LNS_max_turn_ratio, result_list_obj_time)

print(result_list_obj_time)
    
