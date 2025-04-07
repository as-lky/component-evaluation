class INFEASIBLEERROR(Exception):
    pass
class Component:
    def __init__(self, device, taskname, instance, sequence_name):
        self.device = device
        self.taskname = taskname
        self.instance = instance
        self.sequence_name = sequence_name # sequence_name is whole components sequence name 

    def begin(self):
        print(f" {self.__class__.__bases__[0].__name__} Component {self.__class__.__name__} is working ...")
    
    def end(self):
        print(f" {self.__class__.__bases__[0].__name__} Component {self.__class__.__name__} is done.")
    

class LayerConvey:
    ...
# For Component, the parameters from outside are the select args; from front layer are the instance processing result

class Init2Preprocess(LayerConvey): # preprocess for work directory nothing more
    def __init__(self):
        ...
        
class Preprocess2Graphencode(LayerConvey):
    def __init__(self):
        ...
        
class Graphencode2Predict(LayerConvey): # TODO : no generalization
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features):
        self.constraint_features = constraint_features
        self.edge_indices = edge_indices
        self.edge_features = edge_features
        self.variable_features = variable_features
        
        
class Predict2Modify(LayerConvey):
    def __init__(self, b_vars, scores):
        self.b_vars = b_vars
        self.scores = scores
    
    
class Modify2Search(LayerConvey):
    ... # nothing to do
        
class Cantsol(Predict2Modify):
    def __init__(self, logits, select): # a sol which can't be used
        self.logits = logits
        self.select = select
        
class Cansol2M(Predict2Modify):
    def __init__(self, objval, cansol, gap): # a sol which can be used
        self.objval = objval
        self.cansol = cansol
        self.gap = gap


class Cansol2S(Modify2Search):
    def __init__(self, objval, cansol, gap): # a sol which can be used
        self.objval = objval
        self.cansol = cansol
        self.gap = gap

