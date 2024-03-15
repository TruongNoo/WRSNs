# Wireless Sensor Network Simulation

This project simulates a wireless sensor network (WSN) for three cities: Hanoi, Bac Ninh, and Son La. It includes scripts to generate positions of sensor nodes and visualize the network.

## Files

- `ChargePositions.py`: Python script to compute and visualize sensor node positions.
- `Data_WRSN`: Directory containing data files for the WSN simulation.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/TruongNoo/ChargePositions.git
    ```

2. Navigate to the project directory:

    ```bash
    cd your_repository
    ```

3. Ensure you have the required dependencies installed:

    ```bash
    pip install matplotlib numpy pyyaml shapely
    ```

4. Run the `ChargePositions.py` script:

    ```bash
    python ChargePositions.py
    ```

This will generate and visualize the sensor node positions for the specified cities.

## Data

The `Data_WRSN` directory contains the following files:

- `network_scenarios`: Directory containing YAML files describing network scenarios for each city.
- `default.yaml`: Default configuration file for the WSN simulation.

## Dependencies

- matplotlib
- numpy
- pyyaml
- shapely

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
