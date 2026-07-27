from scenario import Scenario
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class Solver:
    def __init__(self, c, scenarios, A_ub=None, b_ub=None):
        self.c = np.asarray(c, dtype=float)
        self.scenarios = scenarios

        # inequality constraints
        self.A_ub = np.asarray(A_ub, dtype=float) if A_ub is not None and len(A_ub) > 0 else None
        self.b_ub = np.asarray(b_ub, dtype=float) if b_ub is not None and len(b_ub) > 0 else None

        # Sanity check on probabilities
        total_prob = sum(s.prob for s in self.scenarios)
        if not np.isclose(total_prob, 1.0):
            raise ValueError(f"Scenario probabilities sum to {total_prob:.4f}, expected 1.0")

    # Solves via deterministic equivalent form
    def solve_deterministic_equivalent(self):
        model = gp.Model("deterministic_equivalent")
        model.Params.OutputFlag = 0

        # First-stage decision variables
        x = model.addMVar(shape=len(self.c), lb=0.0, name="x")

        # First-stage constraints
        if self.A_ub is not None:
            model.addConstr(self.A_ub @ x <= self.b_ub, name="ub_constr")

        # Second-stage decision variables and constraints
        y_vars = []
        expected_recourse_cost = 0

        for i, s in enumerate(self.scenarios):
            y = model.addMVar(shape=len(s.q), lb=0.0, name=f"y_{i}")
            y_vars.append(y)

            # Constraint: W @ y >= h - T @ x  (Standardized to inequality)
            model.addConstr(s.W @ y >= s.h - s.T @ x, name=f"recourse_{i}")

            # Accumulate objective
            expected_recourse_cost += s.prob * (s.q @ y)

        model.setObjective(self.c @ x + expected_recourse_cost, GRB.MINIMIZE)
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return {
                "status": "optimal",
                "optimal_cost": model.ObjVal,
                "x": x.X,
                "y": [y_var.X for y_var in y_vars]
            }
        else:
            raise RuntimeError(f"Optimization failed. Gurobi status: {model.Status}")

    def _solve_subproblem(self, scenario, x_hat):
        sub = gp.Model("subproblem")
        sub.Params.OutputFlag = 0

        y = sub.addMVar(shape=len(scenario.q), lb=0.0)
        rhs = scenario.h - scenario.T @ x_hat

        # Primal constraint: W y >= h - T x
        constr = sub.addConstr(scenario.W @ y >= rhs)
        sub.setObjective(scenario.q @ y, GRB.MINIMIZE)

        sub.optimize()

        if sub.Status == GRB.INFEASIBLE:
            raise RuntimeError("Subproblem is infeasible. Problem lacks complete recourse.")
        elif sub.Status == GRB.UNBOUNDED:
            raise RuntimeError("Subproblem is unbounded.")
        elif sub.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Subproblem failed with status: {sub.Status}")

        return sub.ObjVal, constr.Pi, y.X

    def solve_benders_single_cut(self, max_iter=100, tol=1e-6):
        master = gp.Model("benders_single_cut_master")
        master.Params.OutputFlag = 0

        n_x = len(self.c)
        x = master.addMVar(shape=n_x, lb=0.0, name="x")
        theta = master.addVar(lb=-GRB.INFINITY, name="theta")

        if self.A_ub is not None:
            master.addConstr(self.A_ub @ x <= self.b_ub, name="ub_constr")

        master.setObjective(self.c @ x + theta, GRB.MINIMIZE)

        x_hat = np.zeros(n_x)
        lower_bound, upper_bound = -np.inf, np.inf

        for iteration in range(max_iter):
            expected_recourse = 0
            E_pi_h = 0.0
            E_pi_T = np.zeros(n_x)
            current_y_vals = []

            #solves sub problems for scenarios
            for s in self.scenarios:
                obj_val, pi, y_val = self._solve_subproblem(s, x_hat)
                current_y_vals.append(y_val)
                expected_recourse += s.prob * obj_val
                E_pi_h += s.prob * (pi @ s.h)
                E_pi_T += s.prob * (pi @ s.T)

            #updates upper bound
            current_cost = self.c @ x_hat + expected_recourse
            upper_bound = min(upper_bound, current_cost)

            #addds cut
            master.addConstr(theta >= E_pi_h - E_pi_T @ x, name=f"cut_{iteration}")

            #solves master problem
            master.optimize()
            if master.Status != GRB.OPTIMAL:
                raise RuntimeError("Master problem failed to solve.")

            #updates lower bound and x_hat
            lower_bound = master.ObjVal
            x_hat = x.X

            #checks convergence
            gap = (upper_bound - lower_bound) / max(1e-10, abs(upper_bound))
            if gap <= tol:
                return {
                    "status": "optimal",
                    "optimal_cost": upper_bound,
                    "x": x_hat,
                    "y": current_y_vals,
                    "iterations": iteration + 1
                }

        return {
            "status": "max_iter_reached",
            "optimal_cost": upper_bound,
            "x": x_hat,
            "y": current_y_vals,
            "iterations": max_iter
        }

    def solve_benders_multi_cut(self, max_iter=100, tol=1e-6):
        master = gp.Model("benders_multi_cut_master")
        master.Params.OutputFlag = 0

        n_x = len(self.c)
        n_s = len(self.scenarios)

        x = master.addMVar(shape=n_x, lb=0.0, name="x")
        thetas = master.addMVar(shape=n_s, lb=-GRB.INFINITY, name="theta")

        if self.A_ub is not None:
            master.addConstr(self.A_ub @ x <= self.b_ub, name="ub_constr")

        probs = np.array([s.prob for s in self.scenarios])
        master.setObjective(self.c @ x + probs @ thetas, GRB.MINIMIZE)

        x_hat = np.zeros(n_x)
        lower_bound, upper_bound = -np.inf, np.inf

        for iteration in range(max_iter):
            expected_recourse = 0
            current_y_vals = []
            # solves subproblems and adds a cut per scenario
            for i, s in enumerate(self.scenarios):
                obj_val, pi, y_val = self._solve_subproblem(s, x_hat)
                current_y_vals.append(y_val)
                expected_recourse += s.prob * obj_val

                pi_h = pi @ s.h
                pi_T = pi @ s.T
                master.addConstr(thetas[i] >= pi_h - pi_T @ x, name=f"cut_{iteration}_{i}")

            #update upperbound
            current_cost = self.c @ x_hat + expected_recourse
            upper_bound = min(upper_bound, current_cost)

            #solve master problem
            master.optimize()
            if master.Status != GRB.OPTIMAL:
                raise RuntimeError("Master problem failed to solve.")

            # 4. Update Lower Bound and x_hat
            lower_bound = master.ObjVal
            x_hat = x.X

            #check convergebce
            gap = (upper_bound - lower_bound) / max(1e-10, abs(upper_bound))
            if gap <= tol:
                return {
                    "status": "optimal",
                    "optimal_cost": upper_bound,
                    "x": x_hat,
                    "y": current_y_vals,
                    "iterations": iteration + 1
                }

        return {
            "status": "max_iter_reached",
            "optimal_cost": upper_bound,
            "x": x_hat,
            "y": current_y_vals,
            "iterations": max_iter
        }