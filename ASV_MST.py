import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint, box
from shapely.ops import unary_union
from matplotlib.collections import LineCollection

class Edge:
    def __init__(self, _from, to, weight):
        self.src = _from
        self.dst = to
        self.weight = weight

    def __lt__(self, other):
        return self.weight < other.weight

class Graph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.edges = []
        self.parent = {}
        self.rank = {}
        self.adj_list = {node: [] for node in range(len(nodes))}

    def add_edge(self, src, dst, weight):
        self.edges.append(Edge(src, dst, weight))
        self.adj_list[src].append((dst, weight))
        self.adj_list[dst].append((src, weight))

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

    def kruskal_mst(self):
        self.edges = sorted(self.edges, key=lambda edge: edge.weight)
        mst = []
        for node in range(len(self.nodes)):
            self.parent[node] = node
            self.rank[node] = 0

        for edge in self.edges:
            root1 = self.find(edge.src)
            root2 = self.find(edge.dst)

            if root1 != root2:
                mst.append(edge)
                self.union(root1, root2)

        return mst

def dfs(graph, node, visited, path):
    visited.add(node)
    path.append(node)
    for neighbour, _ in graph.adj_list[node]:
        if neighbour not in visited:
            dfs(graph, neighbour, visited, path)
    path.append(node)  # Return to current node for visual completeness

def create_random_polygon(centroid, irregularity, spikeyness, num_verts):
    irregularity = clip(irregularity, 0, 1) * 2 * np.pi / num_verts
    spikeyness = clip(spikeyness, 0, 1) * centroid[2]
    angles = np.cumsum(np.random.normal(np.pi * 2 / num_verts, irregularity, num_verts))
    angles %= (2 * np.pi)
    radii = np.abs(np.random.normal(centroid[2], spikeyness, num_verts))
    points = [(r * np.cos(a) + centroid[0], r * np.sin(a) + centroid[1]) for r, a in zip(radii, angles)]
    return points

def clip(x, min_val, max_val):
    return max(min(x, max_val), min_val)

def create_convex_hull_polygon(boundary_points):
    points = MultiPoint(boundary_points)
    hull = points.convex_hull
    return Polygon(hull)

def generate_grids(polygon, grid_size):
    min_x, min_y, max_x, max_y = polygon.bounds
    grids = []
    for x in np.arange(min_x, max_x, grid_size):
        for y in np.arange(min_y, max_y, grid_size):
            grid_square = box(x, y, x + grid_size, y + grid_size)
            if polygon.intersects(grid_square):
                grids.append((x + grid_size / 2, y + grid_size / 2))
    return grids

def generate_nodes(grids, grid_size):
    nodes = {}
    for (x, y) in grids:
        neighbors = [
            (x + grid_size, y),
            (x - grid_size, y),
            (x, y + grid_size),
            (x, y - grid_size)
        ]
        for neighbor in neighbors:
            if neighbor in grids:
                if (x, y) not in nodes:
                    nodes[(x, y)] = set()
                nodes[(x, y)].add(neighbor)
    return nodes

def calculate_mst_and_path(boundary, grid_size):
    grids = generate_grids(boundary, grid_size)
    nodes = generate_nodes(grids, grid_size)
    node_list = list(nodes.keys())
    graph = Graph(node_list)
    node_indices = {node: idx for idx, node in enumerate(node_list)}

    for node, neighbors in nodes.items():
        src_idx = node_indices[node]
        for neighbor in neighbors:
            dst_idx = node_indices[neighbor]
            distance = np.hypot(neighbor[0] - node[0], neighbor[1] - node[1])
            graph.add_edge(src_idx, dst_idx, distance)

    mst = graph.kruskal_mst()
    start_node = node_indices[node_list[0]]  # Ensure start_node is a valid index
    visited = set()
    path = []
    dfs(graph, start_node, visited, path)

    return mst, path, grids

def visualize_grid_and_path(boundary, grid_size, mst, path, grids):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw boundary
    x, y = boundary.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Boundary')

    # Draw grid centers
    grid_x, grid_y = zip(*grids)
    ax.scatter(grid_x, grid_y, color='green', s=10, label='Grid Centers')

    # Draw MST edges in blue
    mst_lines = []
    nodes = {i: pos for i, pos in enumerate(grids)}
    for edge in mst:
        x1, y1 = nodes[edge.src]
        x2, y2 = nodes[edge.dst]
        mst_lines.append([(x1, y1), (x2, y2)])
    lc_mst = LineCollection(mst_lines, colors='blue', linewidths=2, label='MST')
    ax.add_collection(lc_mst)

    # Draw Path in red
    path_lines = []
    for i in range(len(path) - 1):
        x1, y1 = nodes[path[i]]
        x2, y2 = nodes[path[i + 1]]
        path_lines.append([(x1, y1), (x2, y2)])
    lc_path = LineCollection(path_lines, colors='red', linewidths=1, label='Trajectory')
    ax.add_collection(lc_path)

    # Adjust plot settings
    ax.autoscale()
    ax.set_aspect('equal')
    plt.legend()
    plt.title("Grid with MST and Trajectory")
    plt.show()

# Generate a random boundary and compute the MST and trajectory
boundary_centroid = (5, 5, 3)
num_boundary_points = 30
boundary_points = create_random_polygon(boundary_centroid, 0.5, 0.2, num_boundary_points)
boundary = create_convex_hull_polygon(boundary_points)
grid_size = 0.6

mst_result, trajectory, grids = calculate_mst_and_path(boundary, grid_size)

print("Edges in MST:", [(edge.src, edge.dst, edge.weight) for edge in mst_result])
print("Trajectory to cover MST:", trajectory)

# Visualize the results
visualize_grid_and_path(boundary, grid_size, mst_result, trajectory, grids)
