import numpy as np
import torch 
from typing import Self, Type, cast
from .mod import Component, Preprocess2Graphencode, Graphencode2Predict
from .help.NEURALDIVING.read_lp import get_a_new2

class Graphencode(Component):
    def __new__(cls, component, *args, **kwargs):
        if component == "bi":
            cls = Bipartite
        elif component == 'bir':
            cls = BipartiteR
        elif component == "tri":
            cls = Tripartite
        elif component == "trir":
            cls = TripartiteR
        elif component == "default":
            cls = Default
        else:
            raise ValueError("Graphencode component type is not defined")

        return super().__new__( cast(type[Self], cls) )
 
 
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)
 
    def work(self) -> Graphencode2Predict: ...

class Bipartite(Graphencode):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        ... # tackle parameters

    def work(self) -> Graphencode2Predict:
        
        self.begin()
        
        constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value = get_a_new2(self.instance)

        self.end()
        
        return Graphencode2Predict( constraint_features, edge_indices, edge_features, variable_features)
        
        
class BipartiteR(Graphencode):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        ... # tackle parameters

    def work(self) -> Graphencode2Predict:
        
        self.begin()
        
        constraint_features, edge_indices, edge_features, variable_features, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type, obj_type, num_to_value = get_a_new2(self.instance, random_feature=True)

        self.end()
        
        return Graphencode2Predict( constraint_features, edge_indices, edge_features, variable_features)
        


class Tripartite(Graphencode):
    # TODO : get tripartite graph code
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        ... # tackle parameters

    def work(self) -> Graphencode2Predict:
        ...


class TripartiteR(Graphencode):
    # TODO : get tripartite graph code
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        ... # tackle parameters

    def work(self) -> Graphencode2Predict:
        ...        
        
        


class Default(Graphencode):
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(component, device, taskname, instance, sequence_name)
        ... # tackle parameters

    def work(self) -> Graphencode2Predict:
        
        self.begin()
        
        # Do nothing...
        
        self.end()
        
        return 0
