# Wireless Rechargeable Sensor Network Simulation

## Description
This project simulates the operation of a Wireless Rechargeable Sensor Network (WRSN) using a combination of Q-learning and heuristic methods for optimizing the charging paths of mobile chargers (MCs). The simulation is visualized using `matplotlib`, and the state of the network is logged and displayed in real-time.

## Features
- **Real-time visualization:** Displays the network's nodes, mobile chargers, and their charging states.
- **Energy logging:** Logs and prints the energy consumption of each node.
- **Q-learning:** Uses Q-learning to optimize the paths of mobile chargers.
- **Non-intersecting circles:** Identifies and removes intersecting circles to improve charging efficiency.

## Prerequisites
- Python 3.6+
- Required Python packages:
  - `matplotlib`
  - `numpy`
  - `shapely`
  - `yaml`
  - `scipy`
  - `pandas`
  - `networkx`

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/TruongNoo/WRSNs.git
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure that the paths in the script (e.g., to the `mc.png` and `bs.png` images) are correct.

## Usage
1. **Setup the Network:**
   - The network configuration is loaded from a YAML file (e.g., `hanoi1000n50.yaml`).
   - The mobile charger configuration is also loaded from a YAML file (e.g., `default.yaml`).

2. **Run the Simulation:**
    ```sh
    python main.py
    ```

3. **Visualize and Log:**
   - The state of the network and mobile chargers is displayed in real-time.
   - Energy levels of nodes and the status of mobile chargers are logged and printed to the console.

## Code Overview
- **Importing Libraries:**
  - Standard libraries for mathematical operations, data handling, and visualization (`math`, `itertools`, `pandas`, `numpy`, `matplotlib`).
  - Specific libraries for network operations and mobile charger management (`networkx`, `physical_env.mc.MobileCharger`, `physical_env.network.NetworkIO`).
  - Custom Q-learning optimizer (`optimizer.q_learning_heuristic`).

- **Functions:**
  - `log(net, mcs, q_learning)`: Logs and visualizes the state of the network and mobile chargers.
  - `print_node_energy(net)`: Prints the energy of each node.
  - `print_state_net(net, mcs)`: Prints the state of the network and mobile chargers.

- **Main Execution:**
  - Loads the network and mobile charger configurations.
  - Initializes the network and mobile chargers.
  - Starts the simulation and logging process.

## Files
- **main.py:** The main script to run the simulation.
- **requirements.txt:** List of required Python packages.
- **physical_env/**: Directory containing network and mobile charger configurations and definitions.
- **optimizer/**: Directory containing the Q-learning heuristic optimizer.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Acknowledge any resources, libraries, or collaborators who contributed to the project.

---
