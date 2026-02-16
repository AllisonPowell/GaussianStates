# Full corrected code for 3D visualization of the oscillator network with all rings, no wraparound on bottom boundary

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_edge(ax, a, b, positions_3d, alpha=0.55, lw=1.0):
    x1, y1, z1 = positions_3d[a]
    x2, y2, z2 = positions_3d[b]
    ax.plot([x1, x2], [y1, y2], [z1, z2], color="black", alpha=alpha, lw=lw)

def connect_1_to_2_cover(ax, lower_nodes, upper_nodes, positions_3d, alpha=0.55, lw=1.0):
    """
    lower_nodes count = m, upper_nodes count = 2m
    Connect lower[j] -> upper[2j], upper[2j+1]  (wrap-safe)
    Guarantees full coverage of all upper nodes exactly once.
    """
    m = len(lower_nodes)
    M = len(upper_nodes)
    assert M == 2*m, f"Expected upper = 2*lower, got {M} vs {m}"

    for j in range(m):
        a = lower_nodes[j]
        b1 = upper_nodes[(2*j) % M]
        b2 = upper_nodes[(2*j + 1) % M]
        draw_edge(ax, a, b1, positions_3d, alpha=alpha, lw=lw)
        draw_edge(ax, a, b2, positions_3d, alpha=alpha, lw=lw)

def connect_1_to_1_scaled(ax, lower_nodes, upper_nodes, positions_3d, alpha=0.55, lw=1.0):
    """
    Connect each lower node to one 'aligned' upper node by index scaling.
    Works for equal sizes and unequal sizes.
    """
    m = len(lower_nodes)
    M = len(upper_nodes)

    for j in range(m):
        a = lower_nodes[j]
        # map j in [0..m-1] to an index in [0..M-1]
        k = int(round(j * M / m)) % M
        b = upper_nodes[k]
        draw_edge(ax, a, b, positions_3d, alpha=alpha, lw=lw)


# Parameters
L = 7
Lh = 5
n_tube = 3
tube_radius = 0.5
top_z = 0.3
bottom_z = -0.5
bdy_len = 2**(L - 1)
bulk_spacing = 0.08
tube_width = 2**(Lh - 1)

# Initialize storage
layer_nodes = []
layer_radii = []
layer_z = []
positions_3d = {}

# Top boundary
top_boundary_nodes = np.arange(0, bdy_len)
layer_nodes.append(top_boundary_nodes)
layer_radii.append(1.2)
layer_z.append(top_z)
for i, node in enumerate(top_boundary_nodes):
    angle = 2 * np.pi * i / bdy_len - 1.5*np.pi/32
    x, y, z = 1.2 * np.cos(angle), 1.2 * np.sin(angle), top_z
    positions_3d[node] = (x, y, z)

# Top bulk layers
bulk_radii_top = [1.1, 0.9]
for i, r in enumerate(bulk_radii_top):
    width = 2**(L - 2 - i)
    start_idx = max(positions_3d) + 1
    nodes = np.arange(start_idx, start_idx + width)
    z = top_z - (i + 1) * bulk_spacing
    layer_nodes.append(nodes)
    layer_radii.append(r)
    layer_z.append(z)
    for j, node in enumerate(nodes):
        if i == 0:
            angle = 2 * np.pi * j / width - np.pi/32
        if i == 1:
            angle = 2 * np.pi * j / width
        positions_3d[node] = (r * np.cos(angle), r * np.sin(angle), z)

# Tube layers
tube_z = np.linspace(top_z - 3 * bulk_spacing, bottom_z + 3 * bulk_spacing, n_tube)
for i, z in enumerate(tube_z):
    start_idx = max(positions_3d) + 1
    nodes = np.arange(start_idx, start_idx + tube_width)
    layer_nodes.append(nodes)
    layer_radii.append(tube_radius)
    layer_z.append(z)
    for j, node in enumerate(nodes):
        angle = 2 * np.pi * j / tube_width
        positions_3d[node] = (tube_radius * np.cos(angle), tube_radius * np.sin(angle), z)

# Bottom bulk layers (reversed)
bulk_radii_bot = [0.9, 1.1]
for i, r in enumerate(bulk_radii_bot):
    width = 2**(L - 2 - (1-i))
    start_idx = max(positions_3d) + 1
    nodes = np.arange(start_idx, start_idx + width)
    z = bottom_z + (len(bulk_radii_bot) - i) * bulk_spacing
    print(bulk_radii_bot[i],z,len(nodes))
    layer_nodes.append(nodes)
    layer_radii.append(r)
    layer_z.append(z)
    for j, node in enumerate(nodes):
        if i == 0:
            angle = 2 * np.pi * j / width
        if i == 1:
            angle = 2 * np.pi * j / width - np.pi/32
        positions_3d[node] = (r * np.cos(angle), r * np.sin(angle), z)

# Bottom boundary
bottom_boundary_nodes = np.arange(max(positions_3d) + 1, max(positions_3d) + 1 + bdy_len)
layer_nodes.append(bottom_boundary_nodes)
layer_radii.append(1.2)
layer_z.append(bottom_z)
for i, node in enumerate(bottom_boundary_nodes):
    angle = 2 * np.pi * i / bdy_len - 1.5*np.pi/32
    positions_3d[node] = (1.2 * np.cos(angle), 1.2 * np.sin(angle), bottom_z)

# Plot all layers, no wraparound on bottom boundary
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.rainbow(np.linspace(0, 1, len(layer_nodes)))
for idx, (nodes, c) in enumerate(zip(layer_nodes, colors)):
    xs = [positions_3d[i][0] for i in nodes]
    ys = [positions_3d[i][1] for i in nodes]
    zs = [positions_3d[i][2] for i in nodes]

    # For bottom boundary (last layer), no wraparound
    #if idx == len(layer_nodes) - 1:
        #for i in range(len(nodes)-1):
            #ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], [zs[i], zs[i+1]], color="black", alpha=0.8)
    #else:
        # For all other layers, wraparound
    xs.append(xs[0])
    ys.append(ys[0])
    zs.append(zs[0])
    ax.plot(xs, ys, zs, '-', color="black", alpha=0.8)

    if idx == 0: #or idx == len(layer_nodes) - 1:
        ax.scatter(xs, ys, zs, s = 20, color = "black")
    else: 
        ax.scatter(xs, ys, zs, s = 20, color = "gray")

    #ax.scatter(xs[:-1] if idx != len(layer_nodes) - 1, else xs, 
               #ys[:-1] if idx != len(layer_nodes) - 1, else ys, 
               #zs[:-1] if idx != len(layer_nodes) - 1, else zs, 
               #s=20, color="black")

# layer_nodes indices: 0..8 correspond to Layers 1..9

# Layer 2 -> Layer 1 (32 -> 64): perfect 1->2 cover
connect_1_to_2_cover(ax, layer_nodes[1], layer_nodes[0], positions_3d, alpha=0.5, lw=1.0)

# Layer 3 -> Layer 2 (16 -> 32): perfect 1->2 cover
connect_1_to_2_cover(ax, layer_nodes[2], layer_nodes[1], positions_3d, alpha=0.5, lw=1.0)

# Layers 4 -> 3, 5 -> 4, 6 -> 5, 7 -> 6 : 1->1
connect_1_to_1_scaled(ax, layer_nodes[3], layer_nodes[2], positions_3d, alpha=0.5, lw=1.0)
connect_1_to_1_scaled(ax, layer_nodes[4], layer_nodes[3], positions_3d, alpha=0.5, lw=1.0)
connect_1_to_1_scaled(ax, layer_nodes[5], layer_nodes[4], positions_3d, alpha=0.5, lw=1.0)
connect_1_to_1_scaled(ax, layer_nodes[6], layer_nodes[5], positions_3d, alpha=0.5, lw=1.0)

# Bottom symmetry:
# Layer 8 -> Layer 7 (32 -> 16): if you want symmetry with Layer 3->2 (2 connections),
# then each node in Layer 8 should connect to ONE node in Layer 7 (since 32 > 16),
# OR flip the "ownership": connect Layer 7 -> Layer 8 as a 1->2 cover.
#
# If you want Layer 7 nodes each connect to 2 nodes in Layer 8 (clean symmetric branching):
connect_1_to_2_cover(ax, layer_nodes[6], layer_nodes[7], positions_3d, alpha=0.5, lw=1.0)

# Layer 9 -> Layer 8 (64 is upper compared to 32): also use 1->2 cover symmetric to Layer 2->1
connect_1_to_2_cover(ax, layer_nodes[7], layer_nodes[8], positions_3d, alpha=0.5, lw=1.0)


ax.set_title("Wormhole Graph")
ax.set_axis_off()
plt.tight_layout()
plt.savefig("plot/wormhole-graph.pdf")
