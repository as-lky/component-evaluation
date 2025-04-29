import random
import time
import os
import numpy as np
from pyscipopt import Model, quicksum

class NetworkResourceAllocationWithBandwidth:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def generate_instance(self):
        channel_demands = self.randint(self.n_channels, self.demand_interval)
        capacities = self.randint(self.n_stations, self.capacity_interval)
        fixed_costs = self.randint(self.n_stations, self.fixed_cost_interval)
        bandwidth_per_station = self.randint(self.n_stations, self.bandwidth_interval)
        
        return {
            'channel_demands': channel_demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'bandwidth_per_station': bandwidth_per_station
        }

    ################# PySCIPOpt modeling #################
    def build_model(self, instance):
        channel_demands = instance['channel_demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        bandwidth_per_station = instance['bandwidth_per_station']

        n_channels = len(channel_demands)
        n_stations = len(capacities)
        
        model = Model("NetworkResourceAllocationWithBandwidth")

        # Decision variables
        StorageUnit = {j: model.addVar(vtype="B", name=f"Station_{j}") for j in range(n_stations)}
        VehicleAllocation = {(i, j): model.addVar(vtype="C", name=f"Bandwidth_{i}_{j}") for i in range(n_channels) for j in range(n_stations)}

        # Objective: Minimize total fixed costs
        objective_expr = quicksum(fixed_costs[j] * StorageUnit[j] for j in range(n_stations))
        model.setObjective(objective_expr, "minimize")

        # Constraints: demand satisfaction
        for i in range(n_channels):
            model.addCons(quicksum(VehicleAllocation[i, j] for j in range(n_stations)) >= channel_demands[i], f"Demand_{i}")

        # Constraints: station bandwidth capacity
        for j in range(n_stations):
            model.addCons(quicksum(VehicleAllocation[i, j] for i in range(n_channels)) <= bandwidth_per_station[j] * StorageUnit[j], f"Capacity_{j}")

        return model

    def save_lp_files(self, output_dir, n_instances):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx in range(n_instances):
            instance = self.generate_instance()
            model = self.build_model(instance)
            lp_filename = os.path.join(output_dir, f"network_resource_allocation_{idx+1}.lp")
            model.writeProblem(lp_filename)
            print(f"Saved: {lp_filename}")

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_channels': 400,     # 控制决策变量数量（主要是 n_channels 和 n_stations 的乘积）
        'n_stations': 120,     # 控制决策变量数量和约束数量（每个 station 和 channel 各产生一条约束）
        'demand_interval': (30, 200),
        'capacity_interval': (200, 2000),
        'fixed_cost_interval': (1200, 3000),
        'bandwidth_interval': (400, 2000),
    }

    n_instances = 5  # 生成 5 个问题
    output_dir = "lp_files"

    network_allocation = NetworkResourceAllocationWithBandwidth(parameters, seed=seed)
    network_allocation.save_lp_files(output_dir, n_instances)