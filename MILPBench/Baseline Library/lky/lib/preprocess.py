import torch
from mod import Component, Init2Preprocess, Preprocess2Graphencode
from typing import Type

class Preprocess(Component): # no usage for now
    def __init__(self, component, *args, **kwargs):
        ... # tackle parameters
    
    def work(self, input: Type[Init2Preprocess]) -> Type[Preprocess2Graphencode]:
        output = Preprocess2Graphencode(input.device, input.taskname, input.instance)        
        return output