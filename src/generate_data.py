import pandas as pd
import numpy as np


def generate_large_dataset(input_file="farmer_data.csv", output_file="farmer_data_large.csv", num_scenarios=500):
    try:
        df = pd.read_csv(input_file, dtype=str)
    except FileNotFoundError:
        print(f"Error: Could not find '{input_file}'. Ensure it is in the same directory.")
        return

    # 1. Keep all non-scenario sections (costs, capacities, base matrices)
    base_df = df[~df['SECTION'].str.startswith('Scenario')].copy()

    # 2. Configure probabilities and random distributions for yields
    prob = 1.0 / num_scenarios
    np.random.seed(42)  # Ensures reproducible results

    # Normal distribution centered around standard yields with realistic variance
    yields_w = np.maximum(0.1, np.random.normal(2.5, 0.6, num_scenarios))  # Wheat
    yields_c = np.maximum(0.1, np.random.normal(3.0, 0.7, num_scenarios))  # Corn
    yields_b = np.maximum(0.1, np.random.normal(20.0, 4.5, num_scenarios))  # Sugar Beets

    new_rows = []

    # 3. Generate dynamic scenarios
    for i in range(1, num_scenarios + 1):
        s_name = f'Scenario{i}'
        new_rows.append({'SECTION': s_name, 'ROW': 'prob', 'VALUES': str(prob)})
        new_rows.append({'SECTION': s_name, 'ROW': 'yield_w', 'VALUES': str(round(yields_w[i - 1], 4))})
        new_rows.append({'SECTION': s_name, 'ROW': 'yield_c', 'VALUES': str(round(yields_c[i - 1], 4))})
        new_rows.append({'SECTION': s_name, 'ROW': 'yield_b', 'VALUES': str(round(yields_b[i - 1], 4))})

    scenarios_df = pd.DataFrame(new_rows)

    # 4. Save to a new large dataset CSV
    final_df = pd.concat([base_df, scenarios_df], ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"Successfully generated '{output_file}' with {num_scenarios} scenarios!")


if __name__ == "__main__":
    # You can scale this up (e.g., 200, 500, 1000) to see how DE vs Benders handles scale
    generate_large_dataset(num_scenarios=500)