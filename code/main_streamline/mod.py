import time

# infeasible error, used when the model can't (predict and repair) a feasible solution
class INFEASIBLEERROR(Exception):
    pass

# the base class for all components in the streamline.
# device: the device that the component is running on
# taskname: the name of the task that the component is running [IS, MVC, MKS, SC]
# instance: the instance the component is running
class Component:
    def __init__(self, device, taskname, instance, sequence_name):
        self.device = device
        self.taskname = taskname
        self.instance = instance
        self.sequence_name = sequence_name # sequence_name is the complete components&hyper-parameters sequence name

    def begin(self):
        self.begin_time = time.time()
        print(f" {self.__class__.__bases__[0].__name__} Component {self.__class__.__name__} is working ...")
    
    def end(self):
        print(f" {self.__class__.__bases__[0].__name__} Component {self.__class__.__name__} is done.")
    

# the base class for contents conveyed between layers in the streamline.
class LayerConvey:
    ...
    
# For Component, the parameters from outside are the selected args; from front layer are the processing contents
class Init2Preprocess(LayerConvey): # preprocess for work directory nothing more
    def __init__(self):
        ...

# preprocess has nothing to convey to graphencode layer
class Preprocess2Graphencode(LayerConvey): 
    def __init__(self):
        ...
        
# actually graph encoding process finished in predict layer
# convey the features to predict layer
class Graphencode2Predict(LayerConvey):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features, obj_features=None, obj_variable_val=None, obj_constraint_val=None, edge_obj_var=None, edge_obj_con=None):
        self.constraint_features = constraint_features
        self.edge_indices = edge_indices
        self.edge_features = edge_features
        self.variable_features = variable_features
        self.obj_features = obj_features
        self.obj_variable_val = obj_variable_val
        self.obj_constraint_val = obj_constraint_val
        self.edge_obj_var = edge_obj_var
        self.edge_obj_con = edge_obj_con
        
# the base class for contents conveyed from predict layer to repair layer 
class Predict2Modify(LayerConvey):
    pass

# the base class for contents conveyed from repair layer to search layer 
class Modify2Search(LayerConvey):
    pass

# convey logits and select to repair layer        
# logits means the predicted values of the instance
# select means the probability that the variable is selected to be the value
class Cantsol(Predict2Modify):
    def __init__(self, logits, select): # a sol which can't be used
        self.logits = logits
        self.select = select
        
# convey objval, cansol and gap to repair layer
# objval means predicted feasible solution's objective value of the instance 
# cansol means the predicted feasible solution of the instance
# gap means the gap between the predicted feasible solution and the optimal solution
class Cansol2M(Predict2Modify):
    def __init__(self, objval, cansol, gap): # a sol which can be used
        self.objval = objval
        self.cansol = cansol
        self.gap = gap

# convey objval, cansol and gap to search layer
# objval means predicted feasible solution's objective value of the instance 
# cansol means the predicted feasible solution of the instance
# gap means the gap between the predicted feasible solution and the optimal solution
class Cansol2S(Modify2Search):
    def __init__(self, objval, cansol, gap): # a sol which can be used
        self.objval = objval
        self.cansol = cansol
        self.gap = gap
