import pandas as pd


def comparison_table(hist_single, hist_multi, det):
    rows = [
        {
            "method": "Benders single-cut",
            "iterations": hist_single["iteration"][-1],
            "time_s": round(hist_single["total_time"], 1),
            "gap_pct": round(hist_single["final_gap"] * 100, 2),
            "objective": round(hist_single["best_upper_bound"][-1]),
        },
        {
            "method": "Benders multi-cut",
            "iterations": hist_multi["iteration"][-1],
            "time_s": round(hist_multi["total_time"], 1),
            "gap_pct": round(hist_multi["final_gap"] * 100, 2),
            "objective": round(hist_multi["best_upper_bound"][-1]),
        },
        {
            "method": "Deterministic (extensive form)",
            "iterations": 1,
            "time_s": round(det["time"], 1),
            "gap_pct": 0.0,
            "objective": round(det["objective"]),
        },
    ]
    return pd.DataFrame(rows)


def facility_plan_df(data, node_list, sol):
    rows = []
    for i in node_list:
        for l in data.facility_sizes:
            if (sol["y"].get((i, l.category)) or 0) > 0.5:
                row = {"node": data.node_names[i], "facility_size": l.descriptor}
                for k in data.commodities:
                    row[k] = round(sol["r"].get((i, k), 0.0), 1)
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["node", "facility_size"] + data.commodities)
    return pd.DataFrame(rows)
