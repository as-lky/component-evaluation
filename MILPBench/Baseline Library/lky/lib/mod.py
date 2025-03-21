class Component:
    def __init__(self, device, taskname, instance, sequence_name):
        self.device = device
        self.taskname = taskname
        self.instance = instance
        self.sequence_name = sequence_name

    def begin(self):
        print(f" {self.__class__.__bases__[0].__name__} Component {self.__class__.__name__} is working ...")
    
    def end(self):
        print(f" {self.__class__.__bases__[0].__name__} Component {self.__class__.__name__} is done.")
    

class LayerConvey:
    ...
# For Component, the parameters from outside are the select args; from front layer are the instance processing result


class Init2Preprocess(LayerConvey):
    def __init__(self):
        ...
        
class Preprocess2Graphencode(LayerConvey):
    def __init__(self):
        ...
        
class Graphencode2Predict(LayerConvey):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features, v_map, v_nodes, c_nodes, b_vars):
        self.constraint_features = constraint_features 
        self.edge_indices = edge_indices
        self.edge_features = edge_features
        self.variable_features = variable_features
        self.v_map = v_map
        self.v_nodes = v_nodes
        self.c_nodes = c_nodes
        self.b_vars = b_vars
        
class Predict2Modify(LayerConvey):
    def __init__(self, b_vars, scores):
        self.b_vars = b_vars
        self.scores = scores
    
    
class Modify2Search(LayerConvey):
    ... # nothing to do
        

class Cansol(Modify2Search):
    def __init__(self, cansol): # a sol which can be used
        self.cansol = cansol
        
class Scores(Modify2Search):
    def __init__(self, b_vars, scores): # predicted scores
        self.b_vars = b_vars
        self.scores = scores
        

