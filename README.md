# ASV_MST_PATH_PLANNING

![image](https://github.com/roboticsmick/ASV_STC_PATH_PLANNING/assets/70121687/485d6e49-4979-4bca-852c-78db3bad8020)

# MST Grid Path Planning

This script generates a Minimum Spanning Tree (MST) for a set of grid nodes within a defined polygon boundary. The process involves generating a random polygon, creating grid points, identifying valid nodes, computing the MST, and visualizing the results.

## Overview

1. **Generate Random Polygon**
    - A random polygon is generated based on a given centroid and parameters.

2. **Create Grid Sets**
    - **Set 1**: Grids of size \(2 \times \text{grid_size}\).
    - **Set 2**: Centers of Set 1 grids.
    - **Set 3**: Centers of subdivided grids (half of grid_size).

3. **Identify Valid Nodes (Set 4)**
    - Nodes from Set 2 are validated based on their surrounding grid points.

4. **Generate MST**
    - Using Kruskal's algorithm, the MST is generated for nodes in Set 4.

5. **Visualization**
    - The boundary, grid points, and MST connections are visualized using `matplotlib`.

## Usage

1. **Install Dependencies**
    ```sh
    pip install numpy matplotlib shapely
    ```

2. **Run the Script**
    ```sh
    python mst_grid_path_planning.py
    ```

This will display the polygon boundary, grid points, and MST connections as orange lines.
