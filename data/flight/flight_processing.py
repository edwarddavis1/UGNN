import os
import numpy as np
from scipy import sparse
import pandas as pd


def get_flight_data():

    base_dir = os.path.dirname(__file__)
    datapath = os.path.join(base_dir, "flight_data")

    # Load data
    T = 36
    As = np.array(
        [sparse.load_npz(os.path.join(datapath, f"As_{t}.npz")) for t in range(T)]
    )
    node_conts = np.load(os.path.join(datapath, "node_continent.npy"))
    node_codes = np.load(
        os.path.join(datapath, "node_airport_codes.npy"), allow_pickle=True
    ).item()
    airports = pd.read_csv(os.path.join(datapath, "airports.csv"))

    # Select EU airports
    euro_nodes_idx = np.where(node_conts == "EU")[0]
    euro_nodes = np.array(list(node_codes.keys()))[euro_nodes_idx]

    # Map airport codes to country labels
    airport_to_country = airports.set_index("ident")["iso_country"].to_dict()
    country_labels = [airport_to_country[code] for code in euro_nodes]

    spatial_node_labels = pd.factorize(np.array(country_labels))[0]
    node_labels = np.tile(spatial_node_labels, T)

    As_euro = np.array([A[euro_nodes_idx, :][:, euro_nodes_idx] for A in As])

    return As_euro, node_labels
