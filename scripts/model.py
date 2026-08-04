import gurobipy as gp
from gurobipy import GRB

# _function that replaces spaces of a st with _
def _safe(name):
    return str(name).replace(" ", "_")

# initialises a gurobi model and turns off the console yap
def _quiet_model(name):
    m = gp.Model(name)
    m.Params.OutputFlag = 0
    return m



#  master problem
def build_master(data, node_list, cut_mode="single"):
    assert cut_mode in ("single", "multi")
    m = _quiet_model("master")

    # decison varibles
    y = {(i, l.category): m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{l.category}") # binary value
         for i in node_list for l in data.facility_sizes} 
    r = {(i, k): m.addVar(lb=0, name=f"r_{i}_{_safe(k)}") #amount of stock to get
         for i in node_list for k in data.commodities}

    # data from .csv files
    cinfo = data.commodity_info #cost
    fixed_cost = gp.quicksum(l.fixed_cost * y[(i, l.category)]
                              for i in node_list for l in data.facility_sizes)
    acquisition_cost = gp.quicksum(cinfo[k].price * r[(i, k)]
                                    for i in node_list for k in data.commodities)

    # tells if in single or multi benders
    if cut_mode == "single":
        theta = m.addVar(lb=0, name="theta")
        m.setObjective(fixed_cost + acquisition_cost + theta, GRB.MINIMIZE)
    else:
        theta = {s["id"]: m.addVar(lb=0, name=f"theta_{s['id']}") for s in data.scenarios}
        expected_theta = gp.quicksum(s["probability"] * theta[s["id"]] for s in data.scenarios)
        m.setObjective(fixed_cost + acquisition_cost + expected_theta, GRB.MINIMIZE)

    # adds capacity cosntr fro each node i
    for i in node_list:
        m.addConstr(
            gp.quicksum(cinfo[k].space * r[(i, k)] for k in data.commodities)
            <= gp.quicksum(l.capacity * y[(i, l.category)] for l in data.facility_sizes),
            name=f"capacity_{i}",
        )
        # at most ine faciility size built at the node i 
        m.addConstr(gp.quicksum(y[(i, l.category)] for l in data.facility_sizes) <= 1,
                    name=f"onefac_{i}")

    m.update()
    return m, y, r, theta

# subproblem 
def build_subproblem(data, commodities, arcs, demand, survival, node_list):
    m = _quiet_model("subproblem")
    cinfo = data.commodity_info

    # creates a free var x, 
    x = {(i, j, k): m.addVar(lb=0, name=f"x_{i}_{j}_{_safe(k)}")
         for (i, j) in arcs for k in commodities}

    # z is excess and w is penalty 
    z = {(i, k): m.addVar(lb=0, name=f"z_{i}_{_safe(k)}") for i in node_list for k in commodities}
    w = {(i, k): m.addVar(lb=0, name=f"w_{i}_{_safe(k)}") for i in node_list for k in commodities}

    transport_cost = gp.quicksum(cinfo[k].transport_cost * dist * x[(i, j, k)]for (i, j), dist in arcs.items() for k in commodities)
    holding_cost = gp.quicksum(cinfo[k].holding * z[(i, k)] for i in node_list for k in commodities)
    penalty_cost = gp.quicksum(cinfo[k].penalty * w[(i, k)] for i in node_list for k in commodities)
    m.setObjective(transport_cost + holding_cost + penalty_cost, GRB.MINIMIZE)

    #creates adjeceny lists, for flow constraint
    in_arcs, out_arcs = {i: [] for i in node_list}, {i: [] for i in node_list}
    for (i, j) in arcs:
        out_arcs[i].append(j)
        in_arcs[j].append(i)

    balance_constr = {}
    
    # flow conservation logic 
    for i in node_list:
        for k in commodities:
            inflow = gp.quicksum(x[(j, i, k)] for j in in_arcs[i])
            outflow = gp.quicksum(x[(i, j, k)] for j in out_arcs[i])
            balance_constr[i,k] = m.addConstr(
                inflow - outflow - z[(i, k)] + w[(i, k)]
                == demand[i][k], #temporary RHS
                name=f"flowcons_{i}_{_safe(k)}",
            )

    m.Params.Method = 1

    return m, balance_constr