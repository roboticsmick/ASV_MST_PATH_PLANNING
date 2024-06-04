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

def generate_grids(polygon, grid_size):
    min_x, min_y, max_x, max_y = polygon.bounds
    grids = []
    for x in np.arange(min_x, max_x, grid_size):
        for y in np.arange(min_y, max_y, grid_size):
            corners = [
                (x, y),
                (x + grid_size, y),
                (x, y + grid_size),
                (x + grid_size, y + grid_size)
            ]
            if all(polygon.contains(Point(corner)) for corner in corners):
                grids.append((x, y))
    return grids

def visualize_boundary_and_grids(boundary, grid_size, grids):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw boundary
    x, y = boundary.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Boundary')

    # Draw grid vertices
    for (x, y) in grids:
        ax.scatter([x, x + grid_size, x, x + grid_size], [y, y, y + grid_size, y + grid_size], color='blue', s=10)

    # Draw grid squares
    for (x, y) in grids:
        square = plt.Rectangle((x, y), grid_size, grid_size, fill=None, edgecolor='gray', linestyle='--', linewidth=0.5)
        ax.add_patch(square)

    # Adjust plot settings
    ax.autoscale()
    ax.set_aspect('equal')
    plt.legend()
    plt.title("Boundary and Grid Vertices")
    plt.show()

# Generate a random boundary
boundary_centroid = (5, 5, 3)
num_boundary_points = 30
boundary_points = create_random_polygon(boundary_centroid, 0.5, 0.2, num_boundary_points)
boundary = create_convex_hull_polygon(boundary_points)
grid_size = 0.6

# Generate grids
grids = generate_grids(boundary, grid_size)

# Visualize the boundary and grids
visualize_boundary_and_grids(boundary, grid_size, grids)
