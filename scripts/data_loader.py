import re
import pandas as pd

DATA_DIR_DEFAULT = "."


class Commodity:
    def __init__(self, name, unit, price, space, transport_cost, penalty, holding):
        self.name = name
        self.unit = unit
        self.price = price
        self.space = space
        self.transport_cost = transport_cost
        self.penalty = penalty
        self.holding = holding


class FacilitySize:
    def __init__(self, category, descriptor, fixed_cost, capacity):
        self.category = category
        self.descriptor = descriptor
        self.fixed_cost = fixed_cost
        self.capacity = capacity


class ProblemData:
    def __init__(self, nodes, node_names, commodities, commodity_info, 
                 facility_sizes, base_arcs, scenarios, hurricanes):
        self.nodes = nodes
        self.node_names = node_names
        self.commodities = commodities
        self.commodity_info = commodity_info
        self.facility_sizes = facility_sizes
        self.base_arcs = base_arcs
        self.scenarios = scenarios
        self.hurricanes = hurricanes


def _parse_int_list(s):
    # Check if it's NaN or an empty string
    if pd.isna(s) or str(s).strip() == "":
        return []
    
    result = []
    parts = str(s).split(",")
    for p in parts:
        if p.strip() != "":
            result.append(int(p))
            
    return result


def _parse_link_list(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    
    result = []
    # Find all pairs of numbers in parentheses using regex
    matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", str(s))
    
    for match in matches:
        node_a = int(match[0])
        node_b = int(match[1])
        result.append((node_a, node_b))
        
    return result


def load_problem_data(data_dir=DATA_DIR_DEFAULT):
    # load all the .csv files
    nodes_df = pd.read_csv(f"{data_dir}/nodes.csv")
    links_df = pd.read_csv(f"{data_dir}/links.csv")
    commodities_df = pd.read_csv(f"{data_dir}/commodities.csv")
    facilities_df = pd.read_csv(f"{data_dir}/facility_sizes.csv")
    hurricanes_df = pd.read_csv(f"{data_dir}/hurricanes.csv")
    scenarios_df = pd.read_csv(f"{data_dir}/scenarios.csv")

    #  get a list of all nodes
    nodes = []
    for index, row in nodes_df.iterrows():
        nodes.append(int(row["node_id"]))
        
    # create dict for nodenames
    node_names = {}
    for index, row in nodes_df.iterrows():
        n_id = int(row["node_id"])
        node_names[n_id] = f"{row['city']}, {row['state']}"

    commodities = []
    commodity_info = {}
    for index, row in commodities_df.iterrows():
        comm_name = row["commodity"]
        commodities.append(comm_name)
        
        c = Commodity(
            name=comm_name, 
            unit=row["unit"],
            price=row["unit_purchase_price_usd"], 
            space=row["unit_space_ft3"],
            transport_cost=row["transport_cost_usd_per_unit_mile"],
            penalty=row["unmet_demand_penalty_usd"], 
            holding=row["holding_cost_usd"]
        )
        commodity_info[comm_name] = c


    facility_sizes = []
    for index, row in facilities_df.iterrows():
        f = FacilitySize(
            category=int(row["size_category"]), 
            descriptor=row["descriptor"], 
            fixed_cost=row["fixed_cost_usd"], 
            capacity=row["capacity_ft3"]
        )
        facility_sizes.append(f)

    base_arcs = {}
    for index, row in links_df.iterrows():
        from_n = int(row["from_node"])
        to_n = int(row["to_node"])
        dist = row["distance_miles"]
        
        base_arcs[(from_n, to_n)] = dist
        base_arcs[(to_n, from_n)] = dist


    hurricanes = {}
    for index, row in hurricanes_df.iterrows():
        h_id = int(row["hurricane_id"])
        
        demands = {
            "Water": row["water_demand_1000gal"],
            "Food": row["food_demand_1000units"],
            "Medical kits": row["medicine_demand_units"]
        }
        
        hurricanes[h_id] = {
            "category": int(row["category"]),
            "landfall_nodes": _parse_int_list(row["landfall_node_ids"]),
            "unusable_links": _parse_link_list(row["unusable_links"]),
            "demand": demands,
            "loss_fraction": row["facility_loss_fraction"],
        }

    scenarios = []
    for index, row in scenarios_df.iterrows():
        h_list = []
        # Check if there is a hurricane 1
        if not pd.isna(row["hurricane_1"]):
            h_list.append(int(row["hurricane_1"]))
        # Check if there is a hurricane 2
        if not pd.isna(row["hurricane_2"]):
            h_list.append(int(row["hurricane_2"]))
            
        s_dict = {
            "id": int(row["scenario_id"]),
            "hurricanes": h_list,
            "probability": row["probability"],
            "type": row["type"],
        }
        scenarios.append(s_dict)

    total_p = 0.0
    for s in scenarios:
        total_p += s["probability"]
        
    if abs(total_p - 1.0) >= 0.001:
        print(f"Warning: scenario probabilities sum to {total_p}, expected 1.0")

    return ProblemData(
        nodes=nodes, 
        node_names=node_names, 
        commodities=commodities,
        commodity_info=commodity_info, 
        facility_sizes=facility_sizes,
        base_arcs=base_arcs, 
        scenarios=scenarios, 
        hurricanes=hurricanes
    )


def build_scenario_network(data: ProblemData, scenario):
    demand = {}
    for i in data.nodes:
        demand[i] = {}
        for k in data.commodities:
            demand[i][k] = 0.0

    survival = {}
    for i in data.nodes:
        survival[i] = 1.0
        
    removed_links = []

    for hid in scenario["hurricanes"]:
        h = data.hurricanes[hid]
        landfall = h["landfall_nodes"]
        
        n_lf = len(landfall)
        if n_lf == 0:
            n_lf = 1
            
        for k in data.commodities:
            share = h["demand"][k] / n_lf
            for i in landfall:
                demand[i][k] += share
                
        for i in landfall:
            new_survival = 1.0 - h["loss_fraction"]
            if new_survival < survival[i]:
                survival[i] = new_survival
                
        for pair in h["unusable_links"]:
            removed_links.append(pair)
            
            reversed_pair = (pair[1], pair[0])
            removed_links.append(reversed_pair)

    arcs = {}
    for ij, dist in data.base_arcs.items():
        if ij not in removed_links:
            arcs[ij] = dist
            
    return demand, survival, arcs