import time
from .data_loader import build_scenario_network
from .model import build_master, solve_subproblem

# calculates the percentage between the lower and upper bound / itter
def gap_trajectory(history):
    best_ub = float("inf")
    gaps = []
    for lb, ub in zip(history["lower_bound"], history["upper_bound"]):
        best_ub = min(best_ub, ub)
        gap = (best_ub - lb) / max(abs(best_ub), 1e-9) * 100
        gaps.append(max(gap, 1e-4))
    return gaps


# calcuates exact y,r based on first stage in mp
def _facility_and_acquisition_cost(data, node_list, y_val, r_val):
    cinfo = data.commodity_info
    fixed = sum(l.fixed_cost * y_val.get((i, l.category), 0.0)
                for i in node_list for l in data.facility_sizes)
    acquisition = sum(cinfo[k].price * r_val.get((i, k), 0.0)
                       for i in node_list for k in data.commodities)
    return fixed, acquisition


def benders(data, node_list, cut_mode="single", tol=1e-4, max_iter=100, verbose=True):
    assert cut_mode in ("single", "multi")

    m, y, r, theta = build_master(data, node_list, cut_mode=cut_mode)

    #creates a dictionary per itter
    history = {
        "iteration": [], "lower_bound": [], "upper_bound": [], "best_upper_bound": [],
        "time_master": [], "time_subproblems": [], "cumulative_time": [], "n_cuts_added": [],
    }

    best_ub = float("inf")
    best_solution = None
    t_start = time.time()
    gap = float("inf")

    #build scenario network function for each scenario
    subproblems = {}

    for s in data.scenarios:
        scenario_id = s["id"]

        demand, survival, arcs = build_scenario_network(data, s)

        subproblems[scenario_id] = {"demand": demand, "survival": survival, "arcs": arcs}

    # subproblem
    for it in range(1, max_iter + 1):
        t0 = time.time()
        m.optimize()
        t_master = time.time() - t0
        lb = m.ObjVal

        y_val = {key: v.X for key, v in y.items()}
        r_val = {key: v.X for key, v in r.items()}

        t0 = time.time()
        scen_obj, scen_duals = {}, {}
        for s in data.scenarios:
            demand, survival, arcs = subproblems[s["id"]]["demand"], subproblems[s["id"]]["survival"], subproblems[s["id"]]["arcs"]
            obj_s, duals_s = solve_subproblem(
                data, data.commodities, arcs, demand, survival, r_val, node_list,
            )
            scen_obj[s["id"]] = obj_s
            scen_duals[s["id"]] = duals_s
        t_sub = time.time() - t0

        expected_recourse = sum(s["probability"] * scen_obj[s["id"]] for s in data.scenarios)
        fixed_cost, acquisition_cost = _facility_and_acquisition_cost(data, node_list, y_val, r_val)
        ub = fixed_cost + acquisition_cost + expected_recourse
        
        # updates ub
        if ub < best_ub - 1e-9:
            best_ub = ub
            best_solution = {
                "y": dict(y_val), "r": dict(r_val),
                "expected_recourse": expected_recourse,
                "fixed_cost": fixed_cost, "acquisition_cost": acquisition_cost,
                "objective": ub,
            }

        history["iteration"].append(it)
        history["lower_bound"].append(lb)
        history["upper_bound"].append(ub)
        history["best_upper_bound"].append(best_ub)
        history["time_master"].append(t_master)
        history["time_subproblems"].append(t_sub)
        history["cumulative_time"].append(time.time() - t_start)

        gap = (best_ub - lb) / max(abs(best_ub), 1e-9)
        if verbose:
            print(f"[{cut_mode:6s}] iter {it:3d}  LB={lb:,.0f}  UB={ub:,.0f}  "
                  f"bestUB={best_ub:,.0f}  gap={gap*100:6.3f}%  "
                  f"t_master={t_master:5.2f}s  t_sub={t_sub:5.2f}s")
            
        # convergece check
        if gap <= tol:
            history["n_cuts_added"].append(0)
            break

        #generates cuts
        if cut_mode == "single":
            intercept = sum(s["probability"] * scen_obj[s["id"]] for s in data.scenarios)
            slope = {}
            for s in data.scenarios:
                p = s["probability"]
                for key, dual in scen_duals[s["id"]].items():
                    slope[key] = slope.get(key, 0.0) + p * dual
            expr = sum(slope[(i, k)] * (r[(i, k)] - r_val.get((i, k), 0.0)) for (i, k) in slope)
            m.addConstr(theta >= intercept + expr, name=f"cut_single_{it}")
            history["n_cuts_added"].append(1)
        else:
            n_added = 0
            for s in data.scenarios:
                sid = s["id"]
                intercept = scen_obj[sid]
                duals_s = scen_duals[sid]
                expr = sum(duals_s[(i, k)] * (r[(i, k)] - r_val.get((i, k), 0.0)) for (i, k) in duals_s)
                m.addConstr(theta[sid] >= intercept + expr, name=f"cut_{sid}_{it}")
                n_added += 1
            history["n_cuts_added"].append(n_added)

        m.update()

    history["converged"] = gap <= tol
    history["total_time"] = time.time() - t_start
    history["final_gap"] = gap
    history["best_solution"] = best_solution
    history["cut_mode"] = cut_mode
    return history
