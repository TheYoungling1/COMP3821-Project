import numpy as np
from numba import njit, config
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import KDTree, ConvexHull
from scipy.stats import linregress
import time
import argparse
import random
import math

# Import from the tiling generation module
from visualize_penrose_rhombus_decagon import PenroseTiling, phi

# Enable Numba disk caching
config.CACHE_DIR = '.numba_cache'

# Constants
TOL = 1e-5

parser = argparse.ArgumentParser(description='Penrose Rhombus Tiling Percolation Analysis')
parser.add_argument('-s', type=int, required=True, metavar='subdivisions',
                    help='How many recursion iterations in Penrose Tiling generation (for the Master Patch)')
parser.add_argument('-t', type=int, default=100, metavar='trials',
                    help='Number of Monte Carlo trials')
parser.add_argument('--Lmin', type=float, default=20.0, 
                    help="Minimum side length (L) of the square frame to analyze.")
parser.add_argument('--Lmax', type=float, default=100.0, 
                    help="Maximum side length (L) of the square frame to analyze.")
parser.add_argument('--Lstep', type=float, default=20.0, 
                    help="Step size for L as it ranges from Lmin to Lmax.")
parser.add_argument('--bt', type=float, default=1.0, 
                    help="Boundary thickness (epsilon) for identifying edge nodes.")

args = parser.parse_args()

# ----------------------------------------------------------------------
# 1. Union-Find Data Structure (from square2.py)
# ----------------------------------------------------------------------
class WeightedQuickUnionUF:
    """
    A class for the Weighted Quick-Union-Find data structure
    with path compression.
    """

    def __init__(self, n):
        """
        Initializes an empty union-find data structure with 'n' sites
        indexed 0 through n-1. Each site is initially in its own component.

        :param n: The number of sites.
        """
        if n <= 0:
            raise ValueError("n must be > 0")
            
        # self.parent[i] = parent of site i
        # Initially, each site is its own parent (root)
        self.parent = list(range(n))
        
        # self.size[i] = number of sites in the tree rooted at i
        # Initially, each tree has size 1
        self.size = [1] * n
        
        # The number of distinct components (or disjoint sets)
        self.count = n

    def get_count(self):
        """
        Returns the number of disjoint sets.
        """
        return self.count

    def _validate(self, p):
        """
        Validates that p is a valid index.
        """
        n = len(self.parent)
        if p < 0 or p >= n:
            raise IndexError(f"index {p} is not between 0 and {n-1}")

    def find(self, p):
        """
        Returns the root (canonical element) of the set containing site 'p'.
        Implements path compression by linking all nodes on the path
        to the root.
        """
        self._validate(p)
        
        # Find the root
        root = p
        while root != self.parent[root]:
            root = self.parent[root]
        
        # Path compression: make every node on path point to root
        while p != root:
            next_p = self.parent[p]
            self.parent[p] = root
            p = next_p
            
        return root

    def connected(self, p, q):
        """
        Returns true if the two sites 'p' and 'q' are in the same component.
        """
        self._validate(p)
        self._validate(q)
        return self.find(p) == self.find(q)

    def union(self, p, q):
        """
        Merges the set containing site 'p' with the set containing site 'q'.
        """
        self._validate(p)
        self._validate(q)
        
        rootP = self.find(p)
        rootQ = self.find(q)

        if rootP == rootQ:
            return  # Already connected

        # This is the "weighted" part:
        # Make the root of the smaller tree point to the root of the larger tree.
        if self.size[rootP] < self.size[rootQ]:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        else:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        
        # A union operation reduces the total number of components by 1
        self.count -= 1


# ----------------------------------------------------------------------
# 2. Penrose Percolation Logic (adapted from square2.py)
# ----------------------------------------------------------------------
class PenrosePercolation:
    def __init__(self, num_nodes, neighbors, left_set, right_set, top_set, bottom_set):
        self.num_nodes = num_nodes
        self.neighbors = neighbors
        self.open_sites = np.zeros(num_nodes, dtype=bool)
        self.num_open = 0
        
        # UF for Left-Right
        self.uf_LR = WeightedQuickUnionUF(num_nodes + 2)
        self.virtual_left = num_nodes
        self.virtual_right = num_nodes + 1
        
        # UF for Top-Bottom
        self.uf_TB = WeightedQuickUnionUF(num_nodes + 2)
        self.virtual_top = num_nodes
        self.virtual_bottom = num_nodes + 1
        
        # Convert sets to Python sets for O(1) lookup if they aren't already
        self.left_set = set(left_set)
        self.right_set = set(right_set)
        self.top_set = set(top_set)
        self.bottom_set = set(bottom_set)

    def open_site(self, site_idx):
        if self.open_sites[site_idx]:
            return
            
        self.open_sites[site_idx] = True
        self.num_open += 1
        
        # Connect to neighbors
        # neighbors is a list of arrays, accessing neighbors[site_idx]
        for nbr in self.neighbors[site_idx]:
            if self.open_sites[nbr]:
                self.uf_LR.union(site_idx, nbr)
                self.uf_TB.union(site_idx, nbr)
                
        # Connect to boundaries
        if site_idx in self.left_set:
            self.uf_LR.union(site_idx, self.virtual_left)
        if site_idx in self.right_set:
            self.uf_LR.union(site_idx, self.virtual_right)
            
        if site_idx in self.top_set:
            self.uf_TB.union(site_idx, self.virtual_top)
        if site_idx in self.bottom_set:
            self.uf_TB.union(site_idx, self.virtual_bottom)

    def percolates_I(self):
        lr = self.uf_LR.connected(self.virtual_left, self.virtual_right)
        tb = self.uf_TB.connected(self.virtual_top, self.virtual_bottom)
        return lr and tb

    def percolates_U(self):
        lr = self.uf_LR.connected(self.virtual_left, self.virtual_right)
        tb = self.uf_TB.connected(self.virtual_top, self.virtual_bottom)
        return lr or tb


class PenrosePercolationStats:
    def __init__(self, num_nodes, neighbors, left_set, right_set, top_set, bottom_set, trials, mode='I'):
        self.trials = trials
        self.results = []
        
        # print(f"  Running {trials} trials for mode {mode}...")
        for i in range(trials):
            sim = PenrosePercolation(num_nodes, neighbors, left_set, right_set, top_set, bottom_set)
            
            # Shuffle all sites to simulate random opening order
            # This avoids random collisions and is efficient
            sites = list(range(num_nodes))
            random.shuffle(sites)
            
            for site in sites:
                sim.open_site(site)
                if mode == 'I':
                    if sim.percolates_I():
                        break
                elif mode == 'U':
                    if sim.percolates_U():
                        break
            
            self.results.append(sim.num_open / num_nodes)

    def mean(self):
        return np.mean(self.results)
    
    def std(self):
        return np.std(self.results)
        
    def confidence_interval(self):
        mean_val = self.mean()
        std_val = self.std()
        margin = (1.96 * std_val) / math.sqrt(self.trials)
        return mean_val - margin, mean_val + margin
    
    def report(self):
        print("-" * 40)
        print(f"mean value of critical value pc = {self.mean(): .6f}")
        print(f"std value of critical value pc = {self.std(): .6f}")
        lo, hi = self.confidence_interval()
        print(f"the 95% confidence interval is {lo} ~ {hi}")
        print("-" * 40)


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

    # print(f"  Collected {len(nodes)} total vertex instances from {len(triangle_vertex_lists)} triangles")

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

    # print(f"  Extracting percolation edges from triangles...")

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


# ----------------------------------------------------------------------
# SUBGRAPH EXTRACTION (Ported from simulation2.py)
# ----------------------------------------------------------------------

def create_subgraph(master_nodes, master_neighbors, inside_original_indices):
    """
    Creates a new, re-indexed sub-graph from a master graph, given
    a list of original node indices to keep.
    
    :param master_nodes: (N, 2) array of all node coordinates
    :param master_neighbors: List of neighbor arrays for all N nodes
    :param inside_original_indices: 1D array of original indices (from 0 to N-1)
                                    that should be included in the new sub-graph.
    :return: (sub_nodes, sub_neighbors, original_to_new_map)
        - sub_nodes: (M, 2) array of coordinates for the new graph (M <= N)
        - sub_neighbors: List of neighbor arrays for the new M-node graph
        - original_to_new_map: A dict {orig_idx: new_idx} for remapping
    """
    
    # Create the new node coordinate array
    sub_nodes = master_nodes[inside_original_indices]
    num_sub_nodes = len(sub_nodes)
    
    # Create the mapping from original index -> new index (0 to M-1)
    original_to_new_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(inside_original_indices)}
    
    # Create the new neighbor list (adjacency list)
    sub_neighbors = [[] for _ in range(num_sub_nodes)]
    
    # Iterate *only* over the nodes we are keeping
    for new_idx, orig_idx in enumerate(inside_original_indices):
        # Look at the original node's neighbors
        for nbr_orig_idx in master_neighbors[orig_idx]:
            # Check if this neighbor is *also* in our sub-graph
            new_nbr_idx = original_to_new_map.get(nbr_orig_idx)
            
            if new_nbr_idx is not None:
                # If yes, add an edge in the new graph
                sub_neighbors[new_idx].append(new_nbr_idx)
                
    # Convert lists to numpy arrays
    sub_neighbors = [np.array(n, dtype=np.int32) for n in sub_neighbors]
    
    return sub_nodes, sub_neighbors, original_to_new_map


def analyze_square_frame(master_nodes, master_neighbors, L, boundary_thickness=1.0):
    """
    Analyzes nodes within a square frame AND builds the corresponding sub-graph.
    
    :return: A dictionary containing the sub-graph and its boundary data.
    """
    
    # 1. Define the square frame boundaries
    # Center the frame at (0,0) which is roughly the center of the sun pattern
    x_center, y_center = 0.0, 0.0
    
    x_min, x_max = x_center - L / 2.0, x_center + L / 2.0
    y_min, y_max = y_center - L / 2.0, y_center + L / 2.0
    
    # 2. Find all nodes *inside* this frame
    nodes = master_nodes
    inside_mask = (nodes[:, 0] >= x_min) & (nodes[:, 0] <= x_max) & \
                  (nodes[:, 1] >= y_min) & (nodes[:, 1] <= y_max)
    
    inside_original_indices = np.where(inside_mask)[0]
    inside_nodes_coords = nodes[inside_original_indices]
    site_count_inside = len(inside_original_indices)
    
    if site_count_inside < 2:
        return {'sub_graph_nodes': np.array([]), 'sub_graph_neighbors': [], 'node_count': 0} # Empty graph

    # 3. Create the sub-graph
    sub_nodes, sub_neighbors, original_to_new_map = create_subgraph(
        master_nodes, 
        master_neighbors, 
        inside_original_indices
    )

    # 4. Find boundary nodes (relative to the *inside* set)
    
    # --- Y-Boundaries (Top and Bottom) ---
    top_mask_relative = (inside_nodes_coords[:, 1] >= y_max - boundary_thickness)
    bottom_mask_relative = (inside_nodes_coords[:, 1] <= y_min + boundary_thickness)
    
    # --- X-Boundaries (Left and Right) ---
    left_mask_relative = (inside_nodes_coords[:, 0] <= x_min + boundary_thickness)
    right_mask_relative = (inside_nodes_coords[:, 0] >= x_max - boundary_thickness)
    
    # Get original indices of boundaries
    top_original = inside_original_indices[top_mask_relative]
    bottom_original = inside_original_indices[bottom_mask_relative]
    left_original = inside_original_indices[left_mask_relative]
    right_original = inside_original_indices[right_mask_relative]
    
    # 5. Remap boundary indices to the *new* sub-graph's indices
    new_top_indices = [original_to_new_map[idx] for idx in top_original]
    new_bottom_indices = [original_to_new_map[idx] for idx in bottom_original]
    new_left_indices = [original_to_new_map[idx] for idx in left_original]
    new_right_indices = [original_to_new_map[idx] for idx in right_original]
    
    total_edges = sum(len(n) for n in sub_neighbors) // 2 # Undirected graph
    
    return {
        'L_value': L,
        'sub_graph_nodes': sub_nodes,
        'sub_graph_neighbors': sub_neighbors,
        'node_count': len(sub_nodes),
        'edge_count': total_edges,
        'top_boundary_nodes': np.array(new_top_indices, dtype=np.int32),
        'bottom_boundary_nodes': np.array(new_bottom_indices, dtype=np.int32),
        'left_boundary_nodes': np.array(new_left_indices, dtype=np.int32),
        'right_boundary_nodes': np.array(new_right_indices, dtype=np.int32)
    }


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

def plot_percolation_stats_IU(L_values, meansI, stdsI, meansU, stdsU, confidence_level=1.96):
    """
    Generates an error bar plot comparing the mean critical probabilities 
    for Intersection (I) and Union (U) percolation vs L.
    """
    plt.figure(figsize=(10, 6))

    # I (Intersection)
    plt.errorbar(
        L_values,
        meansI,
        yerr=stdsI,
        fmt='o-',
        color='blue',
        ecolor='blue',
        capsize=5,
        label='Mean $p_c^I \\pm \\sigma_I$'
    )

    # U (Union) — slightly shifted to avoid marker overlap
    L_shifted = np.array(L_values) * 1.01
    plt.errorbar(
        L_shifted,
        meansU,
        yerr=stdsU,
        fmt='s-',
        color='red',
        ecolor='red',
        capsize=5,
        label='Mean $p_c^U \\pm \\sigma_U$'
    )

    plt.xlabel('Linear System Size ($L$)', fontsize=14)
    plt.ylabel('Mean Critical Probability ($\\bar{p}_c$)', fontsize=14)
    plt.title('Mean $p_c$ vs. System Size ($L$) for I and U Criteria', fontsize=16)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.show()

def plot_extrapolation_IU(L_values, meansI, meansU, exponent=-3/4):
    """
    Plots mean critical probability vs L^(exponent) for I, U, and A=(I+U)/2,
    extrapolates lines, and estimates pc(infinity) for each.
    """
    # Ensure arrays
    L_values = np.asarray(L_values, dtype=float)
    
    meansI = np.asarray(meansI, dtype=float)
    meansU = np.asarray(meansU, dtype=float)
    meansA = 0.5 * (meansI + meansU)

    # 1) Scaling variable
    X_scaling = L_values ** exponent

    plt.figure(figsize=(10, 6))

    results = {}

    # Define plotting configs
    data_sets = {
        'I': {'means': meansI, 'color': 'blue',  'marker': 'o', 'label': 'Intersection ($p_c^I$)'},
        'U': {'means': meansU, 'color': 'red',   'marker': 's', 'label': 'Union ($p_c^U$)'},
        'A': {'means': meansA, 'color': 'green', 'marker': '^', 'label': 'Average ($p_c^A$)'},
    }

    X_plot_min = 0.0
    X_plot_max = float(np.max(X_scaling) * 1.05)
    X_line = np.linspace(X_plot_min, X_plot_max, 100)

    # 2) Fit, plot lines + points, mark intercepts
    for key, data in data_sets.items():
        y = data['means']
        slope, intercept, r_value, p_value, std_err = linregress(X_scaling, y)
        pc_inf = intercept
        results[key] = {'pc_inf': pc_inf, 'R2': r_value**2}

        Y_line = slope * X_line + intercept

        # Fit line
        plt.plot(X_line, Y_line, color=data['color'], linestyle='--',
                 label=f"{data['label']} Fit: $p_c(\\infty)$ = {pc_inf:.5f}")
        # Data points
        plt.plot(X_scaling, y, data['marker'], color=data['color'], markersize=8,
                 label=f"{data['label']} Data $\\bar{{p}}_c(L)$")
        # Intercept marker at X=0
        plt.plot(0, pc_inf, 'x', color=data['color'], markersize=10)

    # 3) Final plot cosmetics
    xlabel_text = f'$L^{{{exponent:.2f}}}$'
    plt.xlabel(xlabel_text, fontsize=14)
    plt.ylabel('Mean Critical Probability ($\\bar{p}_c$)', fontsize=14)
    plt.title('Finite-Size Scaling Extrapolation for I, U, and Average (A)', fontsize=16)

    plt.xlim(X_plot_min - 0.05 * X_plot_max, X_plot_max)
    all_y = np.concatenate([meansI, meansU, meansA])
    plt.ylim(0.95 * float(np.min(all_y)), 1.05 * float(np.max(all_y)))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')

    # 4) Print summary
    print(f"\n--- Extrapolation Results (exponent {exponent:.2f}) ---")
    for key, res in results.items():
        name = {'I':'Intersection', 'U':'Union', 'A':'Average'}[key]
        print(f"{name}: pc(infinity) = {res['pc_inf']:.6f}, R^2 = {res['R2']:.4f}")
    print("-------------------------------------------------------")

    plt.show()

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("PENROSE RHOMBUS TILING PERCOLATION ANALYSIS")
    print("="*60)

    # Number of recursive subdivision iterationsd
    divisions = int(args.s)
    print(f"\nGenerating Master Penrose tiling with {divisions} subdivisions...")
    
    # Create the tiling (Master Patch)
    # We need to ensure the physical scale is large enough to contain the largest requested L
    # The master patch radius is approx scale * 0.5
    # We need scale * 0.5 > Lmax * sqrt(2)/2 (approx)
    # Let's set scale safely large
    required_scale = max(100.0, args.Lmax * 2.5)
    
    tiling = PenroseTiling(
        divisions=divisions,
        base=5,
        scale=required_scale,  # Adjusted scale to fit Lmax
        config={}
    )

    tiling.make_tiling()

    print(f"✓ Generated {len(tiling.triangles)} triangles in master patch")

    # Build neighbor graph for the master patch
    print("\nBuilding master neighbor graph...")
    master_nodes, master_neighbors = build_penrose_neighbor_graph(tiling)
    print(f"✓ Master graph has {len(master_nodes)} unique nodes")
    
    # frame a square based on the grid size L
    # Note: Penrose tiling unit scale is roughly around 100 in the generation code, 
    # but let's verify coordinate ranges to ensure L values make sense
    all_x = master_nodes[:, 0]
    all_y = master_nodes[:, 1]
    min_dim = min(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    
    print(f"  X range: [{all_x.min():.2f}, {all_x.max():.2f}]")
    print(f"  Y range: [{all_y.min():.2f}, {all_y.max():.2f}]")
    print(f"  Max safe square size approx: {min_dim:.2f}")

    # Compute Convex Hull of the master points to ensure frame is strictly inside
    try:
        hull = ConvexHull(master_nodes)
        has_hull = True
    except Exception as e:
        print(f"Warning: Could not compute ConvexHull ({e}). Falling back to bounding box check.")
        has_hull = False
    
    l_values = np.arange(args.Lmin, args.Lmax + 1e-9, args.Lstep)
    valid_l_values = []
    
    meansI = []
    stdsI = []
    meansU = []
    stdsU = []
    
    for l_value in l_values:
        # 1. Bounding box check (fast reject)
        if l_value > min_dim:
            print(f"\nSkipping L={l_value:.2f}: exceeds master patch dimensions ({min_dim:.2f})")
            continue
            
        # 2. Convex Hull check (strict geometric check)
        if has_hull:
            half_L = l_value / 2.0
            corners = np.array([
                [-half_L, -half_L],
                [half_L, -half_L],
                [half_L, half_L],
                [-half_L, half_L]
            ])
            
            # Check if all corners are inside the hull
            # hull.equations: [normal_x, normal_y, offset]
            # dot(normal, point) + offset <= 0 means inside
            # We check: for all equations, for all corners, is the condition met?
            
            # shape: (4, 2) @ (2, n_eq) -> (4, n_eq)
            is_inside = np.all(np.dot(corners, hull.equations[:, :-1].T) + hull.equations[:, -1] <= 1e-5)
            
            if not is_inside:
                print(f"\nSkipping L={l_value:.2f}: Corners extend outside the decagon hull")
                continue
            
        print(f"\n")
        print("="*60)
        print(f"Analyzing square frame size L = {l_value}")
        
        # Extract square frame
        frame_data = analyze_square_frame(
            master_nodes,
            master_neighbors,
            L=l_value,
            boundary_thickness=args.bt
        )
        
        if frame_data['node_count'] == 0:
            print("  No nodes found in this frame. Skipping.")
            continue
        
        N = frame_data['node_count']
        print(f"  Nodes in frame: {N}")
        print(f"  Left boundary: {len(frame_data['left_boundary_nodes'])}")
        print(f"  Right boundary: {len(frame_data['right_boundary_nodes'])}")
        print(f"  Top boundary: {len(frame_data['top_boundary_nodes'])}")
        print(f"  Bottom boundary: {len(frame_data['bottom_boundary_nodes'])}")
        
        if N < 100:
             print("  Warning: Very small system size, results may be unreliable.")

        # Run percolation analysis
        print(f"  Running Monte Carlo percolation analysis ({args.t} trials)...")
        
        # Run Intersection Mode (I)
        stats_I = PenrosePercolationStats(
            num_nodes=N,
            neighbors=frame_data['sub_graph_neighbors'],
            left_set=frame_data['left_boundary_nodes'],
            right_set=frame_data['right_boundary_nodes'],
            top_set=frame_data['top_boundary_nodes'],
            bottom_set=frame_data['bottom_boundary_nodes'],
            trials=args.t,
            mode='I'
        )
        # stats_I.report()
        meansI.append(stats_I.mean())
        stdsI.append(stats_I.std())
        
        # Run Union Mode (U)
        stats_U = PenrosePercolationStats(
            num_nodes=N,
            neighbors=frame_data['sub_graph_neighbors'],
            left_set=frame_data['left_boundary_nodes'],
            right_set=frame_data['right_boundary_nodes'],
            top_set=frame_data['top_boundary_nodes'],
            bottom_set=frame_data['bottom_boundary_nodes'],
            trials=args.t,
            mode='U'
        )
        # stats_U.report()
        meansU.append(stats_U.mean())
        stdsU.append(stats_U.std())
        
        print(f"  pc(I) = {stats_I.mean():.4f} ± {stats_I.std():.4f}")
        print(f"  pc(U) = {stats_U.mean():.4f} ± {stats_U.std():.4f}")
        
        valid_l_values.append(l_value)

    print("\n✓ Analysis complete!")
    
    # Visualize the Master Tiling with Frames
    print("\nGenerating visualization of Master Patch and Frames...")
    visualize_penrose_tiling(tiling, unique_nodes=None, title=f"Penrose Master Patch (S={divisions})")
    ax = plt.gca()
    
    # Draw rectangles for each L
    # Center is (0,0)
    for l_value in valid_l_values:
        rect = plt.Rectangle(
            (-l_value/2, -l_value/2), 
            l_value, l_value, 
            fill=False, 
            edgecolor='red', 
            linewidth=1.5,
            linestyle='--'
        )
        ax.add_patch(rect)
        
    plt.savefig('penrose_frames.png', dpi=150, bbox_inches='tight')
    print("  Saved: penrose_frames.png")
    
    # Plot statistics and extrapolation
    print("\nPlotting summary statistics...")
    if len(valid_l_values) > 0:
        plot_percolation_stats_IU(valid_l_values, meansI, stdsI, meansU, stdsU)
        plot_extrapolation_IU(valid_l_values, meansI, meansU)
    else:
        print("No valid frames analyzed, skipping plots.")
