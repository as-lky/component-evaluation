import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class UDEAP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        return G

    def generate_population_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['population'] = np.random.randint(100, 1000)
            G.nodes[node]['setup_cost'] = np.random.randint(200, 1000)

        for u, v in G.edges:
            G[u][v]['connection_cost'] = (G.nodes[u]['setup_cost'] + G.nodes[v]['setup_cost']) / 3

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_population_costs(G)
        E2 = self.generate_removable_edges(G)
        res = {'G': G, 'E2': E2}
        return res

    ################# LP Model Writer #################
    def write_lp(self, instance, filename="udeap_model.lp"):
        G, E2 = instance['G'], instance['E2']
        
        model = Model("UDEAP")
        
        # Decision Variables
        settlement_vars = {f"S{node}": model.addVar(vtype="B", name=f"S{node}") for node in G.nodes}
        connection_vars = {f"C{u}_{v}": model.addVar(vtype="B", name=f"C{u}_{v}") for u, v in E2}
        funding_vars = {f"F{node}": model.addVar(vtype="B", name=f"F{node}") for node in G.nodes}

        # Objective Function
        objective_expr = quicksum(
            G.nodes[node]['population'] * settlement_vars[f"S{node}"] 
            - G.nodes[node]['setup_cost'] * funding_vars[f"F{node}"]
            for node in G.nodes
        ) - quicksum(
            G[u][v]['connection_cost'] * connection_vars[f"C{u}_{v}"] for u, v in E2
        )
        model.setObjective(objective_expr, "maximize")

        # Budget Constraint
        total_budget_constraint = quicksum(
            G.nodes[node]['setup_cost'] * settlement_vars[f"S{node}"]
            for node in G.nodes
        )
        model.addCons(total_budget_constraint <= self.budget_limit, name="BudgetConstraint")

        # Connection Constraints
        for u, v in E2:
            model.addCons(
                settlement_vars[f"S{u}"] + settlement_vars[f"S{v}"] - connection_vars[f"C{u}_{v}"] <= 1,
                name=f"Connection_{u}_{v}"
            )

        # Funding Constraints
        for node in G.nodes:
            model.addCons(
                funding_vars[f"F{node}"] >= settlement_vars[f"S{node}"],
                name=f"Funding_{node}"
            )

        # Write to LP file
        model.writeProblem(filename)
        print(f"Model written to {filename}")

####################### Main #########################
if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 150,
        'max_n': 300,
        'er_prob': 0.68,
        'alpha': 0.6,
        'budget_limit': 50000,
    }

    udeap = UDEAP(parameters, seed=seed)
    instance = udeap.generate_instance()
    udeap.write_lp(instance, filename="udeap_model.lp")