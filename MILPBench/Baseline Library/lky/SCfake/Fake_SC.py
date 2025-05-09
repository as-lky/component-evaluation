import random
import time
import os
import numpy as np
import scipy
import scipy.sparse
from pyscipopt import Model, quicksum

class SetCoverTelecomIntegration:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_rows * self.n_cols * self.density)

        indices = np.random.choice(self.n_cols, size=nnzrs)
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)
        _, col_nrows = np.unique(indices, return_counts=True)

        indices[:self.n_rows] = np.random.permutation(self.n_rows)

        i = 0
        indptr = [0]
        for n in col_nrows:
            if i >= self.n_rows:
                indices[i:i+n] = np.random.choice(self.n_rows, size=n, replace=False)
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_rows, replace=False)
            i += n
            indptr.append(i)

        c = np.random.randint(self.max_coef, size=self.n_cols) + 1
        service_quality = np.random.normal(loc=self.average_service_quality, scale=5, size=self.n_cols)

        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)
        ).tocsr()

        indices_csr = A.indices
        indptr_csr = A.indptr

        return {
            'c': c,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'service_quality': service_quality
        }

    ################# PySCIPOpt Modeling #################
    def build_model(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        service_quality = instance['service_quality']

        model = Model("SetCoverTelecomIntegration")
        var_names = {}
        quality_vars = {}

        # Create main decision variables
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])
            quality_vars[j] = model.addVar(vtype="B", name=f"q_{j}", obj=-self.quality_weight * service_quality[j])

        # Add constraints to ensure each row is covered
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row+1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"Cover_{row}")

        # Budget constraint
        model.addCons(quicksum(c[j] * var_names[j] for j in range(self.n_cols)) <= self.budget, "BudgetConstraint")

        # Service level constraint
        model.addCons(quicksum(quality_vars[j] for j in range(self.n_cols)) >= self.min_service_quality, "ServiceLevelConstraint")

        # Logical condition constraints
        for j in range(self.n_cols):
            if service_quality[j] > self.quality_threshold:
                model.addCons(var_names[j] <= quicksum(var_names[k] for k in range(max(0, j-1), j+1)), f"LogicalConstraint_{j}")

        # Set objective
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + quicksum(quality_vars[j] for j in range(self.n_cols))
        model.setObjective(objective_expr, "minimize")

        return model

    def save_lp_files(self, output_dir, n_instances):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx in range(n_instances):
            instance = self.generate_instance()
            model = self.build_model(instance)
            lp_filename = os.path.join(output_dir, f"SC_fakemedium_instance{idx}.lp")
            model.writeProblem(lp_filename)
            print(f"Saved: {lp_filename}")

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 100000,  # 控制约束数量（每一行一个覆盖约束）
        'n_cols': 100000,   # 控制决策变量数量（每一列一个决策变量）
        'density': 0.00005, # 控制稀疏度，影响每个约束涉及多少变量
        'max_coef': 112,
        'budget': 1000000,
        'average_service_quality': 0.5,
        'quality_weight': 0.59,
        'min_service_quality': 0,
        'quality_threshold': 0,
    }

    n_instances = 5  # 生成5个实例
    output_dir = "SC_fakemedium_instance"

    set_cover_telecom_problem = SetCoverTelecomIntegration(parameters, seed=seed)
    set_cover_telecom_problem.save_lp_files(output_dir, n_instances)