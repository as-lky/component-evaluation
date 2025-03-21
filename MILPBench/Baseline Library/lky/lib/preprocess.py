import os
from mod import Component, Init2Preprocess, Preprocess2Graphencode
from typing import Type

class Preprocess(Component): # no usage for now
    def __init__(self, device, taskname, instance, sequence_name):
        super.__init__(device, taskname, instance, sequence_name)
    
    def work(self, input: Type[Init2Preprocess]) -> Type[Preprocess2Graphencode]:
        
        print('Preprocess Component is working ...')
        
        if not os.path.isdir(f'./logs'):
            os.mkdir(f'./logs')
        if not os.path.isdir(f'./logs/{self.taskname}'):
            os.mkdir(f'./logs/{self.taskname}')
        if not os.path.isdir(f'./logs/{self.taskname}/{self.sequence_name}'):
            os.mkdir(f'./logs/{self.taskname}/{self.sequence_name}')
        if not os.path.isdir(f'./logs/{self.taskname}/{self.sequence_name}/work'):
            os.mkdir(f'./logs/{self.taskname}/{self.sequence_name}/work')

        print('Preprocess Component is done.')
        
        return Preprocess2Graphencode()