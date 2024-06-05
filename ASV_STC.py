import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint, box, Point
from shapely.ops import unary_union

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

def generate_set1(min_x, min_y, max_x, max_y, grid_size):
    grids = []
    for x in np.arange(min_x, max_x, 2 * grid_size):
        for y in np.arange(min_y, max_y, 2 * grid_size):
            grids.append((x, y))
    return grids

def generate_set2(set1, grid_size):
    nodes = []
    for (x, y) in set1:
        nodes.append((x + grid_size, y + grid_size))
    return nodes

def generate_set3(set2, grid_size):
    subdivided_centers = []
    half_grid_size = grid_size / 2
    for (x, y) in set2:
        subdivided_centers.extend([
            (x - half_grid_size, y - half_grid_size),
            (x + half_grid_size, y - half_grid_size),
            (x - half_grid_size, y + half_grid_size),
            (x + half_grid_size, y + half_grid_size)
        ])
    return subdivided_centers

def filter_inside_polygon(points, polygon):
    return [point for point in points if polygon.contains(Point(point))]

def identify_set4(set2, set3, tolerance):
    set4 = []
    for (x, y) in set2:
        traverse_nodes = [
            (x - tolerance, y - tolerance),
            (x + tolerance, y - tolerance),
            (x - tolerance, y + tolerance),
            (x + tolerance, y + tolerance)
        ]
        if all(any(np.isclose(trav[0], center[0], atol=tolerance) and np.isclose(trav[1], center[1], atol=tolerance) for center in set3) for trav in traverse_nodes):
            set4.append((x, y))
    return set4

def identify_set5(set4, set3, grid_size):
    half_grid_size = grid_size / 2
    set5 = []
    for (x, y) in set4:
        surrounding_points = [
            (x - half_grid_size, y - half_grid_size),
            (x + half_grid_size, y - half_grid_size),
            (x - half_grid_size, y + half_grid_size),
            (x + half_grid_size, y + half_grid_size)
        ]
        set5.extend([point for point in surrounding_points if point in set3])
    return list(set(set5))  # Remove duplicates

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
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def add_edge(self, src, dst, weight):
        self.edges.append(Edge(src, dst, weight))

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
        self.edges = sorted(self.edges, key=lambda edge: (edge.weight, edge.src, edge.dst))
        mst = []
        for edge in self.edges:
            root1 = self.find(edge.src)
            root2 = self.find(edge.dst)

            if root1 != root2:
                mst.append(edge)
                self.union(root1, root2)

        return mst

def generate_mst(set4):
    node_indices = {node: idx for idx, node in enumerate(set4)}
    graph = Graph(list(node_indices.values()))

    for i, (x1, y1) in enumerate(set4):
        for j, (x2, y2) in enumerate(set4):
            if i != j and (x1 == x2 or y1 == y2):  # Only add horizontal and vertical edges
                weight = np.hypot(x2 - x1, y2 - y1)
                graph.add_edge(node_indices[(x1, y1)], node_indices[(x2, y2)], weight)

    mst_edges = graph.kruskal_mst()
    mst = [(set4[edge.src], set4[edge.dst]) for edge in mst_edges]

    return mst

def visualize_sets_and_mst(boundary, set1, set2, set3, set4, set5, mst):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw boundary
    x, y = boundary.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Boundary')

    # Draw Set 1
    if set1:
        set1_x, set1_y = zip(*set1)
        ax.scatter(set1_x, set1_y, color='blue', s=10, label='Grid')

    # Draw Set 2
    if set2:
        set2_x, set2_y = zip(*set2)
        ax.scatter(set2_x, set2_y, color='red', s=30, label='Test nodes')

    # Draw Set 3
    if set3:
        set3_x, set3_y = zip(*set3)
        ax.scatter(set3_x, set3_y, color='green', s=10, label='Test path points')

    # Draw Set 4
    if set4:
        set4_x, set4_y = zip(*set4)
        ax.scatter(set4_x, set4_y, color='purple', s=40, label='Valid nodes')

    # Draw Set 5
    if set5:
        set5_x, set5_y = zip(*set5)
        ax.scatter(set5_x, set5_y, color='orange', s=20, label='Valid Path Points')

    # Draw MST as orange lines
    for (src, dst) in mst:
        ax.plot([src[0], dst[0]], [src[1], dst[1]], color='orange', linewidth=2)

    # Adjust plot settings
    ax.autoscale()
    ax.set_aspect('equal')
    plt.legend()
    plt.title("Boundary, Sets, and MST")
    plt.show()

# Generate a random boundary
boundary_centroid = (5, 5, 3)
num_boundary_points = 30
boundary_points = create_random_polygon(boundary_centroid, 0.5, 0.2, num_boundary_points)
boundary = create_convex_hull_polygon(boundary_points)
grid_size = 0.6
tolerance = 0.4  # Adjust tolerance to be larger for GPS accuracy

# Get the bounds of the polygon
min_x, min_y, max_x, max_y = boundary.bounds

# Generate sets
set1 = generate_set1(min_x, min_y, max_x, max_y, grid_size)
set2 = generate_set2(set1, grid_size)
set3 = generate_set3(set2, grid_size)

# Filter points inside polygon
set1 = filter_inside_polygon(set1, boundary)
set2 = filter_inside_polygon(set2, boundary)
set3 = filter_inside_polygon(set3, boundary)

# Identify Set 4
set4 = identify_set4(set2, set3, tolerance)

# Identify Set 5
set5 = identify_set5(set4, set3, grid_size)

# Generate MST
mst = generate_mst(set4)

# Visualize the sets and MST
visualize_sets_and_mst(boundary, set1, set2, set3, set4, set5, mst)
