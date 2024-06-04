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

def visualize_sets(boundary, set1, set2, set3):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw boundary
    x, y = boundary.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Boundary')

    # Draw Set 1
    if set1:
        set1_x, set1_y = zip(*set1)
        ax.scatter(set1_x, set1_y, color='blue', s=10, label='Set 1')

    # Draw Set 2
    if set2:
        set2_x, set2_y = zip(*set2)
        ax.scatter(set2_x, set2_y, color='red', s=30, label='Set 2 (Nodes)')

    # Draw Set 3
    if set3:
        set3_x, set3_y = zip(*set3)
        ax.scatter(set3_x, set3_y, color='green', s=10, label='Set 3')

    # Adjust plot settings
    ax.autoscale()
    ax.set_aspect('equal')
    plt.legend()
    plt.title("Boundary and Sets")
    plt.show()

# Generate a random boundary
boundary_centroid = (5, 5, 3)
num_boundary_points = 30
boundary_points = create_random_polygon(boundary_centroid, 0.5, 0.2, num_boundary_points)
boundary = create_convex_hull_polygon(boundary_points)
grid_size = 0.6

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

# Visualize the sets
visualize_sets(boundary, set1, set2, set3)
