from __future__ import division
from __future__ import print_function

import os
import re
import glob
import time
import pickle
import random
import argparse
import numpy as np
import gurobipy as gp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .EGAT_models import SpGAT

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # List in tensor data format

    def forward(self, preds, labels):
        """
        preds: logits output values
        labels: labels
        """
        preds = F.softmax(preds, dim=1).to(device)
        eps = 1e-7
        target = self.one_hot(preds.size(1), labels).to(device)
        ce = (-1 * torch.log(preds + eps) * target).to(device)
        floss = (torch.pow((1 - preds), self.gamma) * ce).to(device)
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0), num))
        one[range(labels.size(0)), labels] = 1
        return one



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
