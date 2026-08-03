import time
import pulp
from .data_loader import build_scenario_network


def _safe(name):
    return str(name).replace(" ", "_")


def build_deterministic(data, node_list):
    # initises pulp
    prob = pulp.LpProblem("extensive_form", pulp.LpMinimize)
    cinfo = data.commodity_info

    y = {(i, l.category): pulp.LpVariable(f"y_{i}_{l.category}", cat="Binary")
         for i in node_list for l in data.facility_sizes}
    r = {(i, k): pulp.LpVariable(f"r_{i}_{_safe(k)}", lowBound=0)
         for i in node_list for k in data.commodities}

    fixed_cost = pulp.lpSum(l.fixed_cost * y[(i, l.category)]
                             for i in node_list for l in data.facility_sizes)
    acquisition_cost = pulp.lpSum(cinfo[k].price * r[(i, k)]
                                   for i in node_list for k in data.commodities)

    for i in node_list:
        prob += (
            pulp.lpSum(cinfo[k].space * r[(i, k)] for k in data.commodities)
            <= pulp.lpSum(l.capacity * y[(i, l.category)] for l in data.facility_sizes)
        ), f"capacity_{i}"
        prob += (pulp.lpSum(y[(i, l.category)] for l in data.facility_sizes) <= 1), f"onefac_{i}"

    recourse_cost_terms = []
    for s in data.scenarios:
        sid = s["id"]
        demand, survival, arcs = build_scenario_network(data, s)

        x = {(i, j, k): pulp.LpVariable(f"x_{sid}_{i}_{j}_{_safe(k)}", lowBound=0)
             for (i, j) in arcs for k in data.commodities}
        z = {(i, k): pulp.LpVariable(f"z_{sid}_{i}_{_safe(k)}", lowBound=0)
             for i in node_list for k in data.commodities}
        w = {(i, k): pulp.LpVariable(f"w_{sid}_{i}_{_safe(k)}", lowBound=0)
             for i in node_list for k in data.commodities}

        transport_cost = pulp.lpSum(
            cinfo[k].transport_cost * dist * x[(i, j, k)]
            for (i, j), dist in arcs.items() for k in data.commodities
        )
        holding_cost = pulp.lpSum(cinfo[k].holding * z[(i, k)]
                                   for i in node_list for k in data.commodities)
        penalty_cost = pulp.lpSum(cinfo[k].penalty * w[(i, k)]
                                   for i in node_list for k in data.commodities)
        recourse_cost_terms.append(s["probability"] * (transport_cost + holding_cost + penalty_cost))

        in_arcs, out_arcs = {i: [] for i in node_list}, {i: [] for i in node_list}
        for (i, j) in arcs:
            out_arcs[i].append(j)
            in_arcs[j].append(i)

        for i in node_list:
            for k in data.commodities:
                inflow = pulp.lpSum(x[(j, i, k)] for j in in_arcs[i])
                outflow = pulp.lpSum(x[(i, j, k)] for j in out_arcs[i])
                prob += (
                    inflow + survival[i] * r[(i, k)] - z[(i, k)]
                    == outflow + demand[i][k] - w[(i, k)]
                ), f"flowcons_{sid}_{i}_{_safe(k)}"

    prob += fixed_cost + acquisition_cost + pulp.lpSum(recourse_cost_terms)
    return prob, y, r


def solve_deterministic(data, node_list, time_limit=None, gap_tol=None, msg=False):
    prob, y, r = build_deterministic(data, node_list)

    solver_kwargs = {"msg": msg}
    if time_limit is not None:
        solver_kwargs["timeLimit"] = time_limit
    if gap_tol is not None:
        solver_kwargs["gapRel"] = gap_tol
    solver = pulp.PULP_CBC_CMD(**solver_kwargs)

    t0 = time.time()
    status = prob.solve(solver)
    elapsed = time.time() - t0

    objective = pulp.value(prob.objective)
    y_val = {key: v.value() for key, v in y.items()}
    r_val = {key: (v.value() or 0.0) for key, v in r.items()}

    cinfo = data.commodity_info
    fixed_cost = sum(l.fixed_cost * (y_val.get((i, l.category)) or 0.0)
                      for i in node_list for l in data.facility_sizes)
    acquisition_cost = sum(cinfo[k].price * r_val.get((i, k), 0.0)
                            for i in node_list for k in data.commodities)

    return {
        "status": pulp.LpStatus[status],
        "objective": objective,
        "fixed_cost": fixed_cost,
        "acquisition_cost": acquisition_cost,
        "expected_recourse": objective - fixed_cost - acquisition_cost if objective is not None else None,
        "time": elapsed,
        "y": y_val,
        "r": r_val,
    }
