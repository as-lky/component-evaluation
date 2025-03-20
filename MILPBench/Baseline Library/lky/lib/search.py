import torch
from mod import Component, LayerConvey

class Preprocess2Predict(LayerConvey):
    def __init__(self, log_folder): # log_folder
        self.log_folder = log_folder
        

class Search(Component):
    def Solver(self) : # TODO : check 其接受什么东西
    def work(self, input, *args): 
        output = 3
        return output
    
class Preprocess(Component):
    def preprocess(self, task_type, ins_name_to_read):
        
        
        
        ...
        
        
    def work(self, input, *args): # args: DEVICE TASK_TYPE TASK_NAME 
        if args[0] == "cuda":
            if torch.cuda.is_available():
                self.DEVICE = torch.device("cuda:0")
            else :
                assert False, "CUDA is not available"
        else :
            self.DEVICE = torch.device("cpu")
        