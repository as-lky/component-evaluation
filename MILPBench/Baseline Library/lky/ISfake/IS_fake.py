import os
import sys

# 添加 milp_2181.py 所在路径
sys.path.append(os.path.join(os.path.dirname(__file__), './MIS_easy2/code'))

from milp_2181 import IndependentSet
from pyscipopt import Model, quicksum

def generate_and_save_lp_files(parameters, output_dir, num_instances=3, seed=42123):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_instances):
        print(f"\n🔹 Generating instance {i+1}/{num_instances}...")
    
        # 每次用不同的 seed 保证生成的图不同
        problem = IndependentSet(parameters, seed=seed + i)
        instance = problem.generate_instance()

        graph = instance['graph']
        inequalities = instance['inequalities']
        model = Model(f"IndependentSet_{i}")

        var_names = {}
        for node in graph.nodes:
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        critical_nodes = graph.nodes[:int(graph.number_of_nodes * 0.1)]
        critical_vars = {}
        for k in critical_nodes:
            critical_vars[k] = model.addVar(vtype="B", name=f"y_{k}")

        for count, group in enumerate(inequalities):
            model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"clique_{count}")

        for k in critical_nodes:
            model.addCons(
                quicksum(var_names[i] for i in graph.neighbors[k]) <= len(graph.neighbors[k]) * (1 - critical_vars[k]),
                name=f"indicator_{k}"
            )

        objective_expr = quicksum(var_names[node] for node in graph.nodes) + quicksum(critical_vars[k] for k in critical_nodes)
        model.setObjective(objective_expr, "maximize")

        # 保存 LP 文件
        lp_path = os.path.join(output_dir, f"instance_{i+1}.lp")
        model.writeProblem(lp_path)
        print(f"✅ LP file saved to: {lp_path}")

        # 输出变量和约束数量
        num_vars = model.getNVars()
        num_conss = model.getNConss()
        print(f"📌 Number of variables: {num_vars}")
        print(f"📌 Number of constraints: {num_conss}")

if __name__ == "__main__":
    parameters = {
        'n_nodes': 1000000,
        'edge_probability': 0.1,
        'affinity': 3,
        'graph_type': 'barabasi_albert',
    }

    output_folder = './MIS_easy2/fake'
    generate_and_save_lp_files(parameters, output_folder, num_instances=3)