import numpy as np
from GCN import postion_get

import torch 



# 哪些内容物需要全局提供? 一些路径提供训练测试样本 原始问题 

problem_instance = f'./instance/1'

# component parameters



class Predict(Component): # 提供实例
    def GNNPredict(self, input, *args): # 接受问题实例
        # train
        # infer
        ...    
    def work(self, input, *args):
        output = 2
        return output


class Modify(Component):
    
    def add_radius_constraint(self): # 接受 原问题 和 predict结果 
        ...    
        
    def work(self, input, *args):
    
        
class Search(Component):
    def Solver(self) : # TODO : check 其接受什么东西
    def work(self, input, *args): 
        output = 3
        return output


r, w = Search(), Predict()
print("ASDD")
print(r.work(1), w.work(2))
