import pandas as pd
import numpy as np
import time
import gurobipy as gp

from solver import Solver
from scenario import Scenario


def _load_problem_data(csv_path):
    # format CSV into (c, A_ub, b_ub, scenarios, scenarios_data).
    df = pd.read_csv(csv_path, dtype=str)

    def str_to_floats(val_str):
        return [float(x.strip()) for x in val_str.split(',')]

    c = np.array(str_to_floats(df[df['SECTION'] == 'c']['VALUES'].iloc[0]))
    A_ub = np.array([str_to_floats(df[df['SECTION'] == 'A_ub']['VALUES'].iloc[0])])
    b_ub = np.array([float(df[df['SECTION'] == 'b_ub']['VALUES'].iloc[0])])

    q_base = np.array(str_to_floats(df[df['SECTION'] == 'q']['VALUES'].iloc[0]))
    h_base = np.array(str_to_floats(df[df['SECTION'] == 'h_base']['VALUES'].iloc[0]))
    W_base = np.array([str_to_floats(val) for val in df[df['SECTION'] == 'W']['VALUES']])

    scenarios = []
    scenarios_data = []

    scenario_mask = df['SECTION'].str.startswith('Scenario')
    scenario_rows = df[scenario_mask]
    grouped = {
        s_name: g.set_index('ROW')['VALUES']
        for s_name, g in scenario_rows.groupby('SECTION', sort=False)
    }
    scenario_names = scenario_rows['SECTION'].unique()

    for s_name in scenario_names:
        s_vals = grouped[s_name]

        prob = float(s_vals['prob'])
        yield_w = float(s_vals['yield_w'])
        yield_c = float(s_vals['yield_c'])
        yield_b = float(s_vals['yield_b'])

        T_matrix = np.array([
            [yield_w, 0.0, 0.0],
            [0.0, yield_c, 0.0],
            [0.0, 0.0, yield_b],
            [0.0, 0.0, 0.0],
        ])

        scenarios.append(Scenario(prob=prob, q=q_base, h=h_base, T=T_matrix, W=W_base))
        scenarios_data.append({"name": s_name, "yield_w": yield_w, "yield_c": yield_c, "yield_b": yield_b})

    return c, A_ub, b_ub, scenarios, scenarios_data


def _export_results(res, scenarios_data, method, run_time, output_filename):
    def fmt(val):
        if abs(val) < 0.001:
            return "-"
        return int(round(val))

    x = res['x']
    table_data = [["First Stage", "Area (acres)", fmt(x[0]), fmt(x[1]), fmt(x[2])]]

    for i, s in enumerate(scenarios_data):
        y = res['y'][i]
        name = s.get("name", f"s={i + 1}")

        table_data.append([name, "Yield (T)", fmt(x[0] * s['yield_w']), fmt(x[1] * s['yield_c']), fmt(x[2] * s['yield_b'])])
        table_data.append(["", "Sales (T)", fmt(y[0]), fmt(y[2]), fmt(y[4])])
        table_data.append(["", "Purchase (T)", fmt(y[1]), fmt(y[3]), "-"])

    table_data.append([f"Overall profit: ${-res['optimal_cost']:,.0f}", "", "", "", ""])
    table_data.append([f"Run Time ({method.upper()}):", f"{run_time:.4f} seconds", "", "", ""])

    out_df = pd.DataFrame(table_data, columns=["Stage/Scenario", "Metric", "Wheat", "Corn", "Sugar Beets"])
    out_df.to_excel(output_filename, index=False)



_METHODS = [
    ("de", "Deterministic Equivalent"),
    ("single_cut", "Single-Cut Benders"),
    ("multi_cut", "Multi-Cut Benders"),
]


def run_full_comparison(csv_path, output_prefix="results"):
    c, A_ub, b_ub, scenarios, scenarios_data = _load_problem_data(csv_path)

    results = {}
    total = len(_METHODS)

    model = Solver(c=c, A_ub=A_ub, b_ub=b_ub, scenarios=scenarios)

    for step, (method_key, method_label) in enumerate(_METHODS, start=1):
        print(f"[{step}/{total}] Solving {method_label} ...")

        t_start = time.perf_counter()
        if method_key == "single_cut":
            res = model.solve_benders_single_cut()
        elif method_key in ("benders", "multi_cut"):
            res = model.solve_benders_multi_cut()
        else:
            res = model.solve_deterministic_equivalent()
        run_time = time.perf_counter() - t_start

        iters = res.get("iterations")
        iters_str = f"   Iterations = {iters}" if iters is not None else ""
        print(f"      Objective = {res['optimal_cost']:,.4f}   Runtime = {run_time:.4f}s{iters_str}")

        output_filename = f"{output_prefix}_{method_key}.xlsx"
        _export_results(res, scenarios_data, method_key, run_time, output_filename)
        print(f"      Results saved to {output_filename}\n")

    return results


if __name__ == "__main__":
    csv_file = "farmer_data_large.csv"

    try:
        run_full_comparison(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Run 'generate_data.py' first.")
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")