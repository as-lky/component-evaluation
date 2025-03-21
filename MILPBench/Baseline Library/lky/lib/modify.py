import torch
from typing import Type
from mod import Component, Predict2Modify, Modify2Search
        

class Modify(Component):
    def __new__(cls, component, device, taskname, instance, sequence_name, *args, **kwargs):
        if component == "np":
            return super().__new__(Np, device, taskname, instance, sequence_name, *args, **kwargs)
        else :
            raise ValueError("Modify component type is not defined")
        
    def __init__(self, device, taskname, instance, sequence_name):
        super().__init__(device, taskname, instance, sequence_name)
            
    def work(self, input: Type[Predict2Modify]) -> Type[Modify2Search]:
        pass
        
class Np(Modify): # build a new problem based on the prediction
    def __init__(self, component, device, taskname, instance, sequence_name, *args, **kwargs):
        super().__init__(device, taskname, instance, sequence_name)
        
        if taskname == "IP": 
            dhp = (400, 5)
        elif taskname == "IS":
            dhp = (300, 300) 
        elif taskname == "WA":
            dhp = (0, 600) 
        elif taskname == "CA":
            dhp = (400, 0)
        else :
            dhp = (0, 0)
        
        self.k0 = kwargs.get("k0", dhp[0])
        self.k1 = kwargs.get("k1", dhp[1])
        # tackle parameters    

    def work(self, input: Type[Predict2Modify]) -> Type[Modify2Search]:
        b_vars = input.b_vars
        scores = input.scores
        binary_name = [scores[i][1] for i in b_vars]
        # get a list of (index, VariableName, Prob, -1, type)
        for i in range(len(scores)):
            type = "C"
            if scores[i][1] in binary_name:
                type = 'BINARY'
            scores[i] += [-1, type]

        scores=[x for x in scores if x[4]=='BINARY'] # get binary

        scores.sort(key=lambda x:x[2], reverse=True)

        # fixing variable picked by confidence scores
        count1 = 0
        for i in range(len(scores)):
            if count1 < self.k_1:
                scores[i][3] = 1
                count1 += 1

        scores.sort(key=lambda x: x[2], reverse=False)
        count0 = 0
        for i in range(len(scores)):
            if count0 < self.k_0:
                scores[i][3] = 0
                count0 += 1

        return Modify2Search(scores)
        