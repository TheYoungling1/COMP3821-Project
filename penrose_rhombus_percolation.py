import numpy as np
from numba import njit, config
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import KDTree
import time
import argparse

# Import from the tiling generation module
from visualize_penrose_rhombus import PenroseTiling, phi

# Enable Numba disk caching
config.CACHE_DIR = '.numba_cache'

# Constants
TOL = 1e-5

parser = argparse.ArgumentParser(description='Penrose Rhombus Tiling Percolation Analysis')
parser.add_argument('-s', type=int, required=True, metavar='subdivisions',
                    help='How many recursion iterations in Penrose Tiling generation')
args = parser.parse_args()

def build_penrose_neighbor_graph(tiling):
    """
    Build neighbor graph from Penrose RHOMBUS tiling for percolation analysis.

    CRITICAL: This uses the get_edges_for_percolation() method which:
    - Removes the base edge from both Robinson's triangles

    Args:
        tiling: PenroseTiling instance with tiling generated

    Returns:
        unique_nodes: (N, 2) array of unique vertex coordinates
        neighbors: list of N arrays containing neighbor indices
    """
    # Step 1: Collect ALL vertices from all triangles
    all_vertices = []
    triangle_vertex_lists = []

    for triangle in tiling.triangles:
        verts = triangle.get_all_vertices()
        tri_verts = []
        for v in verts:
            all_vertices.append([v.real, v.imag])
            tri_verts.append(len(all_vertices) - 1)
        triangle_vertex_lists.append(tri_verts)

    nodes = np.array(all_vertices, dtype=np.float64)

    print(f"  Collected {len(nodes)} total vertex instances from {len(triangle_vertex_lists)} triangles")

    # Step 2: Deduplicate vertices using KDTree
    tree = KDTree(nodes)
    pairs = tree.query_pairs(r=TOL)

    # Union-find to merge duplicates
    parent = np.arange(len(nodes))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)

    # Create mapping to unique vertices
    mapping = np.array([find(i) for i in range(len(nodes))])
    unique_ids = np.unique(mapping)
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    node_to_unique = np.array([id_to_idx[mapping[i]] for i in range(len(nodes))])
    unique_nodes = nodes[unique_ids]

    print(f"  Deduplicated to {len(unique_nodes)} unique vertices")

    # Step 3: Collect edges for percolation from each triangle
    # Each triangle represents half a rhombus. We use the get_edges_for_percolation() method

    print(f"  Extracting percolation edges from triangles...")

    # Build a KDTree for fast vertex lookup
    vertex_tree = KDTree(nodes)

    # Collect all percolation edges
    # Finds which vertex in the nodes array corresponds to a given point/coordinate from the edge
    rhombus_edge_set = set()
    for i, triangle in enumerate(tiling.triangles):
        # Get the edges for percolation analysis
        try:
            edges = triangle.get_edges_for_percolation()

            # Convert complex vertices to indices using KDTree
            for v_start, v_end in edges:
                # Query KDTree to find closest vertex (should be exact match within TOL)
                v_start_arr = np.array([[v_start.real, v_start.imag]])
                v_end_arr = np.array([[v_end.real, v_end.imag]])

                # Find the closest vertex in original node list
                _, start_orig_idx = vertex_tree.query(v_start_arr, k=1)
                _, end_orig_idx = vertex_tree.query(v_end_arr, k=1)

                start_orig_idx = start_orig_idx[0]
                end_orig_idx = end_orig_idx[0]

                # Map to unique node indices
                start_idx = node_to_unique[start_orig_idx]
                end_idx = node_to_unique[end_orig_idx]

                if start_idx != end_idx:
                    edge = (min(start_idx, end_idx), max(start_idx, end_idx))
                    rhombus_edge_set.add(edge)
        except Exception as e:
            print(f"  Warning: Error processing triangle {i}: {e}")
            continue

    rhombus_edges = rhombus_edge_set
    print(f"  Collected {len(rhombus_edges)} percolation edges (thin base edges excluded)")

    # Step 4: Build neighbor lists from percolation edges
    neighbors = [[] for _ in range(len(unique_nodes))]
    for i, j in rhombus_edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    neighbors = [np.array(n, dtype=np.int32) for n in neighbors]

    return unique_nodes, neighbors


def neighbors_to_csr(neighbors):
    """Convert neighbor list to CSR format."""
    neighbor_starts = np.zeros(len(neighbors) + 1, dtype=np.int32)
    total = 0
    for i, nbrs in enumerate(neighbors):
        neighbor_starts[i] = total
        total += len(nbrs)
    neighbor_starts[-1] = total

    neighbors_arr = np.zeros(total, dtype=np.int32)
    idx = 0
    for nbrs in neighbors:
        neighbors_arr[idx:idx+len(nbrs)] = nbrs
        idx += len(nbrs)

    return neighbors_arr, neighbor_starts


@njit(cache=True)
def find_cpu(parent, x):
    """Union-find 'find' with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while x != root:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root


@njit(cache=True)
def union_ranked_cpu(parent, rank, a, b):
    """Union-find 'union' with rank optimization."""
    ra = find_cpu(parent, a)
    rb = find_cpu(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


@njit(cache=True)
def newman_ziff_trial_cpu(neighbors_arr, neighbor_starts, N, left_set, right_set):
    """Single Newman-Ziff trial using CPU."""
    parent = np.arange(N+2, dtype=np.int32)
    rank = np.zeros(N+2, dtype=np.int32)

    Lvirt, Rvirt = N, N+1
    sites = np.arange(N)
    np.random.shuffle(sites)

    # Connect boundary nodes to virtual nodes
    for i in left_set:
        union_ranked_cpu(parent, rank, i, Lvirt)
    for i in right_set:
        union_ranked_cpu(parent, rank, i, Rvirt)

    percolates = np.zeros(N, dtype=np.float32)
    open_flags = np.zeros(N, dtype=np.uint8)

    for k in range(N):
        site = sites[k]
        open_flags[site] = 1

        # Connect to open neighbors
        start, end = neighbor_starts[site], neighbor_starts[site+1]
        for j_idx in range(start, end):
            nb = neighbors_arr[j_idx]
            if open_flags[nb]:
                union_ranked_cpu(parent, rank, site, nb)

        # Check if left and right boundaries are connected
        if find_cpu(parent, Lvirt) == find_cpu(parent, Rvirt):
            percolates[k:] = 1.0
            break

    return percolates


def site_crossing_curve_nz_cpu(neighbors_arr, neighbor_starts, left_set, right_set, N, trials=100):
    """CPU Newman-Ziff algorithm for multiple trials."""
    total = np.zeros(N, dtype=np.float32)
    for t in range(trials):
        if (t+1) % 50 == 0:
            print(f"  Progress: {t+1}/{trials} trials")
        total += newman_ziff_trial_cpu(neighbors_arr, neighbor_starts, N, left_set, right_set)
    probs = total / trials
    p_list = np.arange(1, N+1) / N
    return p_list, probs


def estimate_pc_half_height(p_grid, probs):
    """Estimate p_c where crossing probability = 0.5."""
    probs = np.asarray(probs)
    p_grid = np.asarray(p_grid)
    mono = np.maximum.accumulate(probs)
    target = np.clip(0.5, mono.min(), mono.max())
    return float(np.interp(target, mono, p_grid))


def estimate_pc_max_slope(p_grid, probs):
    """Estimate p_c at maximum slope of crossing curve."""
    probs = np.asarray(probs)
    p_grid = np.asarray(p_grid)
    dp = p_grid[2:] - p_grid[:-2]
    dP = probs[2:] - probs[:-2]
    slope = dP / dp
    k = np.argmax(slope)
    pc = (p_grid[k] + p_grid[k+2]) / 2.0
    return float(pc), float(slope[k])


def visualize_penrose_tiling(tiling, unique_nodes=None, title="Penrose Rhombus Tiling"):
    """
    Visualize the Penrose tiling using matplotlib.

    Args:
        tiling: PenroseTiling instance
        unique_nodes: Optional array of nodes to overlay
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)

    # Color scheme for the two triangle types
    color_thin = [0.0, 0.21, 0.95, 0.6]  # Blue for thin triangles
    color_thick = [0.0, 0.53, 1.0, 0.6]  # Light blue for thick triangles

    for triangle in tiling.triangles:
        # Get the 3 vertices of each triangle
        verts = triangle.get_all_vertices()
        xy = [[v.real, v.imag] for v in verts]

        color = color_thin if triangle.shape == "thin" else color_thick
        poly = Polygon(xy, facecolor=color, edgecolor='black', linewidth=0.3)
        ax.add_patch(poly)

    # Optionally overlay vertex points
    if unique_nodes is not None:
        ax.plot(unique_nodes[:, 0], unique_nodes[:, 1], 'k.', markersize=1, alpha=0.3)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)
    ax.autoscale_view()
    plt.tight_layout()
    return fig, ax


def visualize_graph(unique_nodes, neighbors, left_set=None, right_set=None,
                   title="Penrose Rhombus Graph Structure", max_edges=5000):
    """
    Visualize the graph structure with nodes and edges.

    Args:
        unique_nodes: (N, 2) array of node coordinates
        neighbors: list of neighbor arrays for each node
        left_set: indices of left boundary nodes (highlighted in blue)
        right_set: indices of right boundary nodes (highlighted in red)
        title: Plot title
        max_edges: Maximum number of edges to draw (for performance)
    """
    fig, ax = plt.subplots(figsize=(14, 12), dpi=100)

    # Draw edges first (so they appear behind nodes)
    edge_count = 0
    for i, nbrs in enumerate(neighbors):
        if edge_count >= max_edges:
            break
        for j in nbrs:
            if i < j:  # Only draw each edge once
                ax.plot([unique_nodes[i, 0], unique_nodes[j, 0]],
                       [unique_nodes[i, 1], unique_nodes[j, 1]],
                       'gray', linewidth=0.3, alpha=0.5, zorder=1)
                edge_count += 1

    # Draw all nodes
    ax.plot(unique_nodes[:, 0], unique_nodes[:, 1], 'ko',
           markersize=3, alpha=0.6, zorder=2, label='Nodes')

    # Highlight boundary nodes
    if left_set is not None and len(left_set) > 0:
        ax.plot(unique_nodes[left_set, 0], unique_nodes[left_set, 1],
               'bo', markersize=6, alpha=0.8, zorder=3, label='Left boundary')

    if right_set is not None and len(right_set) > 0:
        ax.plot(unique_nodes[right_set, 0], unique_nodes[right_set, 1],
               'ro', markersize=6, alpha=0.8, zorder=3, label='Right boundary')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=16, pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)

    # Add info text
    info_text = f"Nodes: {len(unique_nodes)}\nEdges shown: {min(edge_count, max_edges)}"
    if edge_count >= max_edges:
        info_text += f"\n(limited for performance)"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig, ax


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("PENROSE RHOMBUS TILING PERCOLATION ANALYSIS")
    print("="*60)

    # Number of recursive subdivision iterationsd
    divisions = int(args.s)
    print(f"\nGenerating Penrose tiling with {divisions} subdivisions...")

    # Create the tiling
    tiling = PenroseTiling(
        divisions=divisions,
        base=5,
        scale=100,  # Scale for proper sizing
        config={}
    )

    tiling.make_tiling()

    print(f"✓ Generated {len(tiling.triangles)} triangles")

    # Build neighbor graph
    print("\nBuilding neighbor graph...")
    unique_nodes, neighbors = build_penrose_neighbor_graph(tiling)

    # Visualize the tiling
    print("\nVisualizing tiling...")
    visualize_penrose_tiling(tiling, unique_nodes,
                            title=f"Penrose Rhombus Tiling ({divisions} subdivisions)")
    plt.savefig('penrose_rhombus_tiling.png', dpi=150, bbox_inches='tight')
    print("  Saved: penrose_rhombus_tiling.png")
    plt.show()

    # Compute statistics
    avg_neighbors = np.mean([len(n) for n in neighbors])
    print(f"✓ Graph has {len(unique_nodes)} unique nodes")
    print(f"  Average degree: {avg_neighbors:.2f}")

    # Count tile types
    stats = tiling.get_statistics()
    print(f"  Thin triangles: {stats['thin']}")
    print(f"  Thick triangles: {stats['thick']}")
    print(f"  Ratio thick/thin: {stats['thick_to_thin_ratio']:.4f} (expected φ ≈ {phi:.4f})")

    # Convert to CSR format
    print("\nConverting to CSR format...")
    neighbors_arr, neighbor_starts = neighbors_to_csr(neighbors)

    # Define boundaries (left/right)
    all_x = unique_nodes[:, 0]
    xmin, xmax = all_x.min(), all_x.max()
    tol = 1e-5
    left_set = np.where(all_x <= xmin + tol)[0].astype(np.int32)
    right_set = np.where(all_x >= xmax - tol)[0].astype(np.int32)
    N = len(unique_nodes)

    print(f"✓ Left boundary: {len(left_set)} nodes")
    print(f"  Right boundary: {len(right_set)} nodes")

    # Visualize the graph structure
    print("\nVisualizing graph structure...")
    visualize_graph(unique_nodes, neighbors, left_set, right_set,
                   title=f"Penrose Rhombus Graph ({divisions} subdivisions, {N} nodes)",
                   max_edges=10000)
    plt.savefig('penrose_rhombus_graph.png', dpi=150, bbox_inches='tight')
    print("  Saved: penrose_rhombus_graph.png")
    plt.show()

    # Run percolation analysis
    print("\nRunning Newman-Ziff percolation analysis...")
    trials = 1000  # More trials for better statistics
    t0 = time.time()
    p_list, probs = site_crossing_curve_nz_cpu(neighbors_arr, neighbor_starts,
                                                left_set, right_set, N, trials=trials)
    t1 = time.time()
    print(f"✓ Completed {trials} trials in {t1-t0:.2f}s")

    # Estimate critical threshold
    pc_half = estimate_pc_half_height(p_list, probs)
    pc_slope, max_slope = estimate_pc_max_slope(p_list, probs)

    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Critical threshold (P=0.5):     p_c = {pc_half:.4f}")
    print(f"Critical threshold (max slope): p_c = {pc_slope:.4f}")
    print(f"Maximum slope:                  {max_slope:.2f}")
    print("="*60)

    # Plot crossing probability curve
    plt.figure(figsize=(10, 6))
    plt.plot(p_list, probs, 'b-', linewidth=2, label='Crossing probability')
    plt.axvline(pc_half, color='r', linestyle='--', label=f'p_c (P=0.5) = {pc_half:.4f}')
    plt.axvline(pc_slope, color='g', linestyle='--', label=f'p_c (max slope) = {pc_slope:.4f}')
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel("p (site open probability)", fontsize=12)
    plt.ylabel("Left-right crossing probability", fontsize=12)
    plt.title("Site Percolation on Penrose Rhombus Tiling", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('penrose_rhombus_percolation_curve.png', dpi=150, bbox_inches='tight')
    print("  Saved: penrose_rhombus_percolation_curve.png")
    plt.show()

    print("\n✓ Analysis complete!")
