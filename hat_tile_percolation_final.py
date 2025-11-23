from turtle import right
from matplotlib.font_manager import is_opentype_cff_font
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import linregress
import argparse

##################################### HATTILING BUILD ##################################
# Constants
sqrt3 = math.sqrt(3)
ident = [1, 0, 0, 0, 1, 0]
PI = math.pi

# Colours
cols = {
    'H1': [153/255, 100/255, 1],
    'H': [229/255, 205/255, 1],
    'T': [224/255, 224/255, 224/255],
    'P': [250/255, 250/255, 250/255],
    'F': [255/255, 255/255, 198/255],
    'edge': [0, 0, 0]
}

# Grid systems
def pt(x, y):
    return {'x': x, 'y': y}

def hexPt(x, y):
    return pt(x + 0.5*y, (sqrt3/2)*y)

# Affine transform functions
def inv(T):
    det = T[0]*T[4] - T[1]*T[3]
    return [T[4]/det, -T[1]/det, (T[1]*T[5]-T[2]*T[4])/det,
            -T[3]/det, T[0]/det, (T[2]*T[3]-T[0]*T[5])/det]

def mul(A, B):
    return [A[0]*B[0] + A[1]*B[3], 
            A[0]*B[1] + A[1]*B[4],
            A[0]*B[2] + A[1]*B[5] + A[2],
            A[3]*B[0] + A[4]*B[3], 
            A[3]*B[1] + A[4]*B[4],
            A[3]*B[2] + A[4]*B[5] + A[5]]

def padd(p, q):
    return {'x': p['x'] + q['x'], 'y': p['y'] + q['y']}

def psub(p, q):
    return {'x': p['x'] - q['x'], 'y': p['y'] - q['y']}

def trot(ang):
    c = math.cos(ang)
    s = math.sin(ang)
    return [c, -s, 0, s, c, 0]

def ttrans(tx, ty):
    return [1, 0, tx, 0, 1, ty]

def rotAbout(p, ang):
    return mul(ttrans(p['x'], p['y']), mul(trot(ang), ttrans(-p['x'], -p['y'])))

def transPt(M, P):
    return pt(M[0]*P['x'] + M[1]*P['y'] + M[2], M[3]*P['x'] + M[4]*P['y'] + M[5])

def matchSeg(p, q):
    return [q['x']-p['x'], p['y']-q['y'], p['x'], q['y']-p['y'], q['x']-p['x'], p['y']]

def matchTwo(p1, q1, p2, q2):
    return mul(matchSeg(p2, q2), inv(matchSeg(p1, q1)))

def intersect(p1, q1, p2, q2):
    d = (q2['y'] - p2['y']) * (q1['x'] - p1['x']) - (q2['x'] - p2['x']) * (q1['y'] - p1['y'])
    uA = ((q2['x'] - p2['x']) * (p1['y'] - p2['y']) - (q2['y'] - p2['y']) * (p1['x'] - p2['x'])) / d
    return pt(p1['x'] + uA * (q1['x'] - p1['x']), p1['y'] + uA * (q1['y'] - p1['y']))

# Hat outline
hat_outline = [
    hexPt(0, 0), hexPt(-1,-1), hexPt(0,-2), hexPt(2,-2),
    hexPt(2,-1), hexPt(4,-2), hexPt(5,-1), hexPt(4, 0),
    hexPt(3, 0), hexPt(2, 2), hexPt(0, 3), hexPt(0, 2),
    hexPt(-1, 2)]

# Tile classes
class HatTile:
    def __init__(self, label):
        self.label = label
        self.shape = hat_outline
    def draw(self, S, level):
        drawPolygon(hat_outline, S, cols[self.label], cols['edge'])
        return

class MetaTile:
    def __init__(self, shape, width):
        self.shape = shape 
        self.width = width
        self.children = [] 
    def addChild(self, T, geom):
        self.children.append({'T': T, 'geom': geom})
        return
    def evalChild(self, n, i):
        return transPt(self.children[n]['T'], self.children[n]['geom'].shape[i])
    def draw(self, S, level):
        if level > 0:
            for g in self.children:
                g['geom'].draw(mul(S, g['T']), level - 1)
        else:
            drawPolygon(self.shape, S, None, 'black')
        return
    def recentre(self):
        cx = sum(p['x'] for p in self.shape) / len(self.shape)
        cy = sum(p['y'] for p in self.shape) / len(self.shape)
        tr = pt(-cx, -cy)
        self.shape = [padd(p, tr) for p in self.shape]
        M = ttrans(-cx, -cy)
        for ch in self.children:
            ch['T'] = mul(M, ch['T'])
        return

def drawPolygon(shape, T, f=cols['H'], e=cols['edge']):
    polygon = [transPt(T, p) for p in shape]
    ax.fill([p['x'] for p in polygon], [p['y'] for p in polygon],
            facecolor=f, edgecolor=e, linewidth=1)
    return

# Initialize tiles
H1_hat = HatTile('H1')
H_hat = HatTile('H')
T_hat = HatTile('T')
P_hat = HatTile('P')
F_hat = HatTile('F')

def H_init():
    H_outline = [
        pt(0, 0), pt(4, 0), pt(4.5, sqrt3/2),
        pt(2.5, 5 * sqrt3/2), pt(1.5, 5 * sqrt3/2), pt(-0.5, sqrt3/2)
    ]
    meta = MetaTile(H_outline, 2)
    meta.addChild(matchTwo(hat_outline[5], hat_outline[7], H_outline[5], H_outline[0]), H_hat)
    meta.addChild(matchTwo(hat_outline[9], hat_outline[11], H_outline[1], H_outline[2]), H_hat)
    meta.addChild(matchTwo(hat_outline[5], hat_outline[7], H_outline[3], H_outline[4]), H_hat)
    meta.addChild(mul(ttrans(2.5, sqrt3/2), mul([-0.5,-sqrt3/2,0,sqrt3/2,-0.5,0], [0.5,0,0,0,-0.5,0])), H1_hat)
    return meta

def T_init():
    T_outline = [pt(0, 0), pt(3, 0), pt(1.5, 3 * sqrt3/2)]
    meta = MetaTile(T_outline, 2)
    meta.addChild([0.5, 0, 0.5, 0, 0.5, sqrt3/2], T_hat)
    return meta

def P_init():
    P_outline = [pt(0, 0), pt(4, 0), pt(3, 2 * sqrt3/2), pt(-1, 2 * sqrt3/2)]
    meta = MetaTile(P_outline, 2)
    meta.addChild([0.5, 0, 1.5, 0, 0.5, sqrt3/2], P_hat)
    meta.addChild(mul(ttrans(0, 2 * sqrt3/2), mul([0.5, sqrt3/2, 0, -sqrt3/2, 0.5, 0], [0.5, 0.0, 0.0, 0.0, 0.5, 0.0])), P_hat)
    return meta

def F_init():
    F_outline = [pt(0, 0), pt(3, 0), pt(3.5, sqrt3/2), pt(3, 2 * sqrt3/2), pt(-1, 2 * sqrt3/2)]
    meta = MetaTile(F_outline, 2)
    meta.addChild([0.5, 0, 1.5, 0, 0.5, sqrt3/2], F_hat)
    meta.addChild(mul(ttrans(0, 2 * sqrt3/2), mul([0.5, sqrt3/2, 0, -sqrt3/2, 0.5, 0], [0.5, 0.0, 0.0, 0.0, 0.5, 0.0])), F_hat)
    return meta

def constructPatch(H,T,P,F):
    rules = [
        ['H'], [0, 0, 'P', 2], [1, 0, 'H', 2], [2, 0, 'P', 2], [3, 0, 'H', 2],
        [4, 4, 'P', 2], [0, 4, 'F', 3], [2, 4, 'F', 3], [4, 1, 3, 2, 'F', 0],
        [8, 3, 'H', 0], [9, 2, 'P', 0], [10, 2, 'H', 0], [11, 4, 'P', 2],
        [12, 0, 'H', 2], [13, 0, 'F', 3], [14, 2, 'F', 1], [15, 3, 'H', 4],
        [8, 2, 'F', 1], [17, 3, 'H', 0], [18, 2, 'P', 0], [19, 2, 'H', 2],
        [20, 4, 'F', 3], [20, 0, 'P', 2], [22, 0, 'H', 2], [23, 4, 'F', 3],
        [23, 0, 'F', 3], [16, 0, 'P', 2], [9, 4, 0, 2, 'T', 2], [4, 0, 'F', 3]
    ]
    
    ret = MetaTile([], H.width)
    shapes = {'H': H, 'T': T, 'P': P, 'F': F}
    
    for r in rules:
        if len(r) == 1:
            ret.addChild(ident, shapes[r[0]])
        elif len(r) == 4:
            poly = ret.children[r[0]]['geom'].shape
            T = ret.children[r[0]]['T']
            P = transPt(T, poly[(r[1]+1)%len(poly)])
            Q = transPt(T, poly[r[1]])
            nshp = shapes[r[2]]
            npoly = nshp.shape
            ret.addChild(matchTwo(npoly[r[3]], npoly[(r[3]+1)%len(npoly)], P, Q), nshp)
        else:
            chP = ret.children[r[0]]
            chQ = ret.children[r[2]]
            P = transPt(chQ['T'], chQ['geom'].shape[r[3]])
            Q = transPt(chP['T'], chP['geom'].shape[r[1]])
            nshp = shapes[r[4]]
            npoly = nshp.shape
            ret.addChild(matchTwo(npoly[r[5]], npoly[(r[5]+1)%len(npoly)], P, Q), nshp)
    return ret

def constructMetatiles(patch):
    bps1 = patch.evalChild(8, 2)
    bps2 = patch.evalChild(21, 2)
    rbps = transPt(rotAbout(bps1, -2.0*PI/3.0), bps2)
    p72 = patch.evalChild(7, 2)
    p252 = patch.evalChild(25, 2)
    llc = intersect(bps1, rbps, patch.evalChild(6, 2), p72)
    w = psub(patch.evalChild(6, 2), llc)
    
    new_H_outline = [llc, bps1]
    w = transPt(trot(-PI/3), w)
    new_H_outline.append(padd(new_H_outline[1], w))
    new_H_outline.append(patch.evalChild(14, 2))
    w = transPt(trot(-PI/3), w)
    new_H_outline.append(psub(new_H_outline[3], w))
    new_H_outline.append(patch.evalChild(6, 2))
    
    new_H = MetaTile(new_H_outline, patch.width * 2)
    for ch in [0, 9, 16, 27, 26, 6, 1, 8, 10, 15]:
        new_H.addChild(patch.children[ch]['T'], patch.children[ch]['geom'])
    
    new_P_outline = [p72, padd(p72, psub(bps1, llc)), bps1, llc]
    new_P = MetaTile(new_P_outline, patch.width * 2)
    for ch in [7, 2, 3, 4, 28]:
        new_P.addChild(patch.children[ch]['T'], patch.children[ch]['geom'])
    
    new_F_outline = [bps2, patch.evalChild(24, 2), patch.evalChild(25, 0), p252, padd(p252, psub(llc, bps1))]
    new_F = MetaTile(new_F_outline, patch.width * 2)
    for ch in [21, 20, 22, 23, 24, 25]:
        new_F.addChild(patch.children[ch]['T'], patch.children[ch]['geom'])
    
    AAA = new_H_outline[2]
    BBB = padd(new_H_outline[1], psub(new_H_outline[4], new_H_outline[5]))
    CCC = transPt(rotAbout(BBB, -PI/3), AAA)
    new_T_outline = [BBB, CCC, AAA]
    new_T = MetaTile(new_T_outline, patch.width * 2)
    new_T.addChild(patch.children[11]['T'], patch.children[11]['geom'])
    
    new_H.recentre()
    new_P.recentre()
    new_F.recentre()
    new_T.recentre()
    
    return [new_H, new_T, new_P, new_F]

def build_neighbor_graph_fast(patch, level=0):
    estimated_nodes = min(1000 * (4 ** level), 50000000)
    nodes = np.empty((estimated_nodes, 2), dtype=np.float64)
    node_count = [0]
    
    def _collect_nodes(patch, S, level):
        if level > 0 and hasattr(patch, "children"):
            for g in patch.children:
                _collect_nodes(g['geom'], mul(S, g['T']), level-1)
        else:
            for p in patch.shape:
                pt_screen = transPt(S, p)
                if node_count[0] < len(nodes):
                    nodes[node_count[0]] = [pt_screen['x'], pt_screen['y']]
                    node_count[0] += 1
    
    _collect_nodes(patch, [1,0,0,0,1,0], level)
    nodes = nodes[:node_count[0]]
    
    tree = KDTree(nodes)
    tol = 1e-5
    pairs = tree.query_pairs(r=tol)
    
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
    
    mapping = np.array([find(i) for i in range(len(nodes))])
    unique_ids = np.unique(mapping)
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    node_to_unique = np.array([id_to_idx[mapping[i]] for i in range(len(nodes))])
    unique_nodes = nodes[unique_ids]
    
    edges_set = set()
    node_idx = 0
    
    def _collect_edges(patch, S, level):
        nonlocal node_idx
        # Only recurse if it has children AND level > 0
        if level > 0 and hasattr(patch, "children"):
            for g in patch.children:
                _collect_edges(g['geom'], mul(S, g['T']), level-1)
        else:
            # base case: patch.shape exists
            n = len(patch.shape)
            base_idx = node_idx
            for i in range(n):
                idx1 = node_to_unique[base_idx + i]
                idx2 = node_to_unique[base_idx + (i+1)%n]
                if idx1 != idx2:
                    edges_set.add((min(idx1, idx2), max(idx1, idx2)))
            node_idx += n
    
    _collect_edges(patch, [1,0,0,0,1,0], level)
    
    neighbors = [[] for _ in range(len(unique_nodes))]
    for i, j in edges_set:
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


##################################### SQUARE FRAME HATTILING BUILD ##################################
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
    x_min, x_max = 80 - L / 2.0, 80 + L / 2.0
    y_min, y_max = -L / 2.0, L / 2.0
    
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
    # (Assuming create_subgraph is defined elsewhere and works correctly)
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
    left_original = inside_original_indices[left_mask_relative]   # NEW
    right_original = inside_original_indices[right_mask_relative] # NEW
    
    # 5. Remap boundary indices to the *new* sub-graph's indices
    new_top_indices = [original_to_new_map[idx] for idx in top_original]
    new_bottom_indices = [original_to_new_map[idx] for idx in bottom_original]
    new_left_indices = [original_to_new_map[idx] for idx in left_original]     # NEW
    new_right_indices = [original_to_new_map[idx] for idx in right_original]   # NEW
    
    total_edges = sum(len(n) for n in sub_neighbors) // 2 # Undirected graph
    
    return {
        'L_value': L,
        'sub_graph_nodes': sub_nodes,
        'sub_graph_neighbors': sub_neighbors,
        'node_count': len(sub_nodes),
        'edge_count': total_edges,
        'top_boundary_nodes': np.array(new_top_indices, dtype=np.int32),
        'bottom_boundary_nodes': np.array(new_bottom_indices, dtype=np.int32),
        'left_boundary_nodes': np.array(new_left_indices, dtype=np.int32),     # NEW
        'right_boundary_nodes': np.array(new_right_indices, dtype=np.int32)   # NEW
    }

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
    L_shifted = L_values * 1.01
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


# --------------------------------------------------------------------------------------

### 2. Extrapolation Plot (`plot_extrapolation_RD`)
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



#######################################################################################
# weighted quick union-find
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
        
class HatPercolationTB():
    def __init__(self, nodes, neighbours, top_set, bottom_set) -> None:
        
        self.N = len(nodes)
        
        self.neighbors_arr, self.neighbor_starts = neighbors_to_csr(neighbours)

        
        self.top_set = top_set
        self.bottom_set = bottom_set
        
        # other variables
        self.sites = np.array([0 for _ in range(self.N)])
        self.wqfGrid = WeightedQuickUnionUF(self.N + 2) # virtual top and bottom and left and right
        self.wqfFull = WeightedQuickUnionUF(self.N + 1) # top
        
        self.virtualTop = self.N
        self.virtualBottom = self.N + 1
        self.openSite = 0
    
    def open_site(self, idx: int) -> None:
        # idx: 0 to N - 1
        self.validState(idx)
        
        if (self.isOpen(idx)):
            return 
        
        self.sites[idx] = 1
        self.openSite += 1
        
        # connect to neighbours
        
        # top
        if idx in self.top_set:
            self.wqfFull.union(self.N, idx)
            self.wqfGrid.union(self.N, idx)
        
        # bottom
        if idx in self.bottom_set:
            self.wqfGrid.union(self.N+1, idx) 

        # neighbours
        start, end = self.neighbor_starts[idx], self.neighbor_starts[idx + 1]
        for j in range(start, end):
            neigh = self.neighbors_arr[j]
            if (self.isOpen(neigh)):
                self.wqfGrid.union(idx, neigh)               
                self.wqfFull.union(idx, neigh)               
        
    def isOpen(self, idx: int) -> bool:
        self.validState(idx)
        return self.sites[idx]
    
    def isFull(self, idx: int) -> bool:
        self.validState(idx)
        return self.wqfFull.connected(self.virtualTop, idx)
    
    def validState(self, idx: int):
        if not self.inHat(idx):
            raise IndexError("index is out of bounds")
        
    def inHat(self, idx: int) -> bool:
        return (idx >= 0 and idx < self.N)
    
    def percolates(self, ) -> bool:
        return self.wqfGrid.connected(self.virtualTop, self.virtualBottom)
    
class percolationStatsTB:
    def __init__(self, nodes, neighbours, top_set, bottom_set, trials: int):
                
        if (trials <= 0):
            raise ValueError("trials must be postive integer")

        self.trialCount = trials
        self.trialResults = []
        
        for i in range(self.trialCount):
            simulator = HatPercolationTB(nodes, neighbours, top_set, bottom_set)
            while (not simulator.percolates()):
                idx = random.randint(0, (simulator.N - 1))
                simulator.open_site(idx)
                
            openSites = simulator.openSite
            result = openSites / (simulator.N)
            self.trialResults.append(result)
            
            
            
    def trials_mean(self,):
        return np.mean(self.trialResults)

    def trials_std(self, ):
        return np.std(self.trialResults)
    
    def trails_confidence_interval(self,):
        return self.trials_mean() - (1.96 * self.trials_std()) / math.sqrt(self.trialCount), self.trials_mean() + (1.96 * self.trials_std()) / math.sqrt(self.trialCount)
    
    def report(self, ):

        print("="*60)
        print("STATS REPORT") 
        print("="*60)

        print(f"mean value of critical value pc = {self.trials_mean(): .6f}")
        print(f"std value of critical value pc = {self.trials_std(): .6f}")
        lo, hi = self.trails_confidence_interval()
        print(f"the 95% confidence interval is {lo} ~ {hi}")
        print("="*60)


class HatPercolationLR():
    def __init__(self, nodes, neighbours, left_set, right_set) -> None:
        
        self.N = len(nodes)
        
        self.neighbors_arr, self.neighbor_starts = neighbors_to_csr(neighbours)

        
        self.left_set = left_set
        self.right_set = right_set
        
        # other variables
        self.sites = np.array([0 for _ in range(self.N)])
        self.wqfGrid = WeightedQuickUnionUF(self.N + 2) # left and right
        self.wqfFull = WeightedQuickUnionUF(self.N + 1) # left
        
        self.virtualLeft = self.N
        self.virtualRight = self.N + 1
        self.openSite = 0
    
    def open_site(self, idx: int) -> None:
        # idx: 0 to N - 1
        self.validState(idx)
        
        if (self.isOpen(idx)):
            return 
        
        self.sites[idx] = 1
        self.openSite += 1
        
        # connect to neighbours
        
        # top
        if idx in self.left_set:
            self.wqfFull.union(self.N, idx)
            self.wqfGrid.union(self.N, idx)
        
        # bottom
        if idx in self.right_set:
            self.wqfGrid.union(self.N+1, idx) 

        # neighbours
        start, end = self.neighbor_starts[idx], self.neighbor_starts[idx + 1]
        for j in range(start, end):
            neigh = self.neighbors_arr[j]
            if (self.isOpen(neigh)):
                self.wqfGrid.union(idx, neigh)               
                self.wqfFull.union(idx, neigh)               
        
    def isOpen(self, idx: int) -> bool:
        self.validState(idx)
        return self.sites[idx]
    
    def isFull(self, idx: int) -> bool:
        self.validState(idx)
        return self.wqfFull.connected(self.virtualLeft, idx)
    
    def validState(self, idx: int):
        if not self.inHat(idx):
            raise IndexError("index is out of bounds")
        
    def inHat(self, idx: int) -> bool:
        return (idx >= 0 and idx < self.N)
    
    def percolates(self, ) -> bool:
        return self.wqfGrid.connected(self.virtualLeft, self.virtualRight)
    
class percolationStatsLR:
    def __init__(self, nodes, neighbours, left_set, right_set, trials: int):
                
        if (trials <= 0):
            raise ValueError("trials must be postive integer")

        self.trialCount = trials
        self.trialResults = []
        
        for i in range(self.trialCount):
            simulator = HatPercolationLR(nodes, neighbours, left_set, right_set)
            while (not simulator.percolates()):
                idx = random.randint(0, (simulator.N - 1))
                simulator.open_site(idx)
                
            openSites = simulator.openSite
            result = openSites / (simulator.N)
            self.trialResults.append(result)
            
            
            
    def trials_mean(self,):
        return np.mean(self.trialResults)

    def trials_std(self, ):
        return np.std(self.trialResults)
    
    def trails_confidence_interval(self,):
        return self.trials_mean() - (1.96 * self.trials_std()) / math.sqrt(self.trialCount), self.trials_mean() + (1.96 * self.trials_std()) / math.sqrt(self.trialCount)
    
    def report(self, ):

        print("="*60)
        print("STATS REPORT") 
        print("="*60)

        print(f"mean value of critical value pc = {self.trials_mean(): .6f}")
        print(f"std value of critical value pc = {self.trials_std(): .6f}")
        lo, hi = self.trails_confidence_interval()
        print(f"the 95% confidence interval is {lo} ~ {hi}")
        print("="*60)

class HatPercolationI():
    def __init__(self, nodes, neighbours, top_set, bottom_set, left_set, right_set) -> None:
        
        self.N = len(nodes)
        
        self.neighbors_arr, self.neighbor_starts = neighbors_to_csr(neighbours)

        
        self.top_set = top_set
        self.bottom_set = bottom_set
        self.left_set = left_set
        self.right_set = right_set
        
        # other variables
        self.sites = np.array([0 for _ in range(self.N)])
        self.wqfGrid_LR = WeightedQuickUnionUF(self.N + 2)
        self.wqfGrid_TB = WeightedQuickUnionUF(self.N + 2)
        
        self.virtualTop = self.N
        self.virtualBottom = self.N + 1
        
        self.virtualLeft = self.N
        self.virtualRight = self.N + 1
        
        self.openSite = 0
    
    def open_site(self, idx: int) -> None:
        # idx: 0 to N - 1
        self.validState(idx)
        
        if (self.isOpen(idx)):
            return 
        
        self.sites[idx] = 1
        self.openSite += 1
        
        # connect to neighbours
        
        # top
        if idx in self.top_set:
            self.wqfGrid_TB.union(self.virtualTop, idx)
        
        # bottom
        if idx in self.bottom_set:
            self.wqfGrid_TB.union(self.virtualBottom, idx) 
            
        # leftmost
        if idx in self.left_set:
            self.wqfGrid_LR.union(self.virtualLeft, idx)
            
        # rightmost
        if idx in self.right_set:
            self.wqfGrid_LR.union(self.virtualRight, idx)

        # neighbours
        start, end = self.neighbor_starts[idx], self.neighbor_starts[idx + 1]
        for j in range(start, end):
            neigh = self.neighbors_arr[j]
            if (self.isOpen(neigh)):
                self.wqfGrid_LR.union(idx, neigh) 
                self.wqfGrid_TB.union(idx, neigh)                             
                                            
        
    def isOpen(self, idx: int) -> bool:
        self.validState(idx)
        return self.sites[idx]
    
    
    def validState(self, idx: int):
        if not self.inHat(idx):
            raise IndexError("index is out of bounds")
        
    def inHat(self, idx: int) -> bool:
        return (idx >= 0 and idx < self.N)
    
    def percolates(self, ) -> bool:
        return self.wqfGrid_TB.connected(self.virtualTop, self.virtualBottom) and self.wqfGrid_LR.connected(self.virtualLeft, self.virtualRight)
    
class percolationStatsI():
    def __init__(self, nodes, neighbours, top_set, bottom_set, left_set, right_set, trials: int):
                
        if (trials <= 0):
            raise ValueError("trials must be postive integer")

        self.trialCount = trials
        self.trialResults = []
        
        for i in range(self.trialCount):
            simulator = HatPercolationI(nodes, neighbours, top_set, bottom_set, left_set, right_set)
            while (not simulator.percolates()):
                idx = random.randint(0, (simulator.N - 1))
                simulator.open_site(idx)
            
            
            openSites = simulator.openSite
            # print(f"number of opensites are: {openSites}")
            result = openSites / (simulator.N)
            self.trialResults.append(result)
            
            
            
    def trials_mean(self,):
        return np.mean(self.trialResults)

    def trials_std(self, ):
        return np.std(self.trialResults)
    
    def trails_confidence_interval(self,):
        return self.trials_mean() - (1.96 * self.trials_std()) / math.sqrt(self.trialCount), self.trials_mean() + (1.96 * self.trials_std()) / math.sqrt(self.trialCount)
    
    def report(self, ):

        print("="*60)
        print("STATS REPORT") 
        print("="*60)

        print(f"mean value of critical value pc = {self.trials_mean(): .6f}")
        print(f"std value of critical value pc = {self.trials_std(): .6f}")
        lo, hi = self.trails_confidence_interval()
        print(f"the 95% confidence interval is {lo} ~ {hi}")
        print("="*60)

class HatPercolationU():
    def __init__(self, nodes, neighbours, top_set, bottom_set, left_set, right_set) -> None:
        
        self.N = len(nodes)
        
        self.neighbors_arr, self.neighbor_starts = neighbors_to_csr(neighbours)

        
        self.top_set = top_set
        self.bottom_set = bottom_set
        self.left_set = left_set
        self.right_set = right_set
        
        # other variables
        self.sites = np.array([0 for _ in range(self.N)])
        self.wqfGrid_LR = WeightedQuickUnionUF(self.N + 2)
        self.wqfGrid_TB = WeightedQuickUnionUF(self.N + 2)
        
        self.virtualTop = self.N
        self.virtualBottom = self.N + 1
        
        self.virtualLeft = self.N
        self.virtualRight = self.N + 1
        
        self.openSite = 0
    
    def open_site(self, idx: int) -> None:
        # idx: 0 to N - 1
        self.validState(idx)
        
        if (self.isOpen(idx)):
            return 
        
        self.sites[idx] = 1
        self.openSite += 1
        
        # connect to neighbours
        
        # top
        if idx in self.top_set:
            self.wqfGrid_TB.union(self.virtualTop, idx)
        
        # bottom
        if idx in self.bottom_set:
            self.wqfGrid_TB.union(self.virtualBottom, idx) 
            
        # leftmost
        if idx in self.left_set:
            self.wqfGrid_LR.union(self.virtualLeft, idx)
            
        # rightmost
        if idx in self.right_set:
            self.wqfGrid_LR.union(self.virtualRight, idx)

        # neighbours
        start, end = self.neighbor_starts[idx], self.neighbor_starts[idx + 1]
        for j in range(start, end):
            neigh = self.neighbors_arr[j]
            if (self.isOpen(neigh)):
                self.wqfGrid_LR.union(idx, neigh) 
                self.wqfGrid_TB.union(idx, neigh)                             
                                            
        
    def isOpen(self, idx: int) -> bool:
        self.validState(idx)
        return self.sites[idx]
    
    
    def validState(self, idx: int):
        if not self.inHat(idx):
            raise IndexError("index is out of bounds")
        
    def inHat(self, idx: int) -> bool:
        return (idx >= 0 and idx < self.N)
    
    def percolates(self, ) -> bool:
        return self.wqfGrid_TB.connected(self.virtualTop, self.virtualBottom) or self.wqfGrid_LR.connected(self.virtualLeft, self.virtualRight)

class percolationStatsU():
    def __init__(self, nodes, neighbours, top_set, bottom_set, left_set, right_set, trials: int):
                
        if (trials <= 0):
            raise ValueError("trials must be postive integer")

        self.trialCount = trials
        self.trialResults = []
        
        for i in range(self.trialCount):
            simulator = HatPercolationU(nodes, neighbours, top_set, bottom_set, left_set, right_set)
            while (not simulator.percolates()):
                idx = random.randint(0, (simulator.N - 1))
                simulator.open_site(idx)
                
            openSites = simulator.openSite
            result = openSites / (simulator.N)
            self.trialResults.append(result)
            
            
            
    def trials_mean(self,):
        return np.mean(self.trialResults)

    def trials_std(self, ):
        return np.std(self.trialResults)
    
    def trails_confidence_interval(self,):
        return self.trials_mean() - (1.96 * self.trials_std()) / math.sqrt(self.trialCount), self.trials_mean() + (1.96 * self.trials_std()) / math.sqrt(self.trialCount)
    
    def report(self, ):

        print("="*60)
        print("STATS REPORT") 
        print("="*60)

        print(f"mean value of critical value pc = {self.trials_mean(): .6f}")
        print(f"std value of critical value pc = {self.trials_std(): .6f}")
        lo, hi = self.trails_confidence_interval()
        print(f"the 95% confidence interval is {lo} ~ {hi}")
        print("="*60)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Monte Carlo simulation for HatTile."
    )
    
    parser.add_argument(
        '--r', 
        type=int, 
        default=2, 
        help="The number of recursion level."
    )
    
    parser.add_argument(
        '--t', 
        type=int, 
        default=1000, 
        help="The number of Monte Carlo trials to perform."
    )
    
    parser.add_argument(
        '--Lmin', type=float, default=50.0, 
        help="Minimum side length (L) of the square frame to analyze."
    )
    parser.add_argument(
        '--Lmax', type=float, default=100.0, 
        help="Maximum side length (L) of the square frame to analyze."
    )
    parser.add_argument(
        '--Lstep', type=float, default=10.0, 
        help="Step size for L as it ranges from Lmin to Lmax."
    )
    parser.add_argument(
        '--bt', type=float, default=1, 
        help="Boundary thickness (epsilon) for identifying edge nodes."
    )
    

    args = parser.parse_args()
    recursion = args.r
    trials = args.t
    
    meansI = []
    stdsI = []
    
    meansU = []
    stdsU = []
    
    # Build a large and fixed patch
    print("="*6)
    print("Building Large Patch")
    base_tiles = [H_init(), T_init(), P_init(), F_init()]
    cur_tiles  = base_tiles
    patch = None
    for _ in range(recursion):
        patch = constructPatch(*cur_tiles)
        cur_tiles = constructMetatiles(patch)
    
    master_nodes, master_neighbors = build_neighbor_graph_fast(patch, level=recursion+1)
    print("INFORMATION:")
    print(f"number of nodes: {len(master_neighbors)}")
    print("="*6)
    
    # frame a square based on the grid size L
    l_values = np.arange(args.Lmin, args.Lmax + 1e-9, args.Lstep)
    
    for l_value in l_values:
        print(f"\n")
        print("="*60)
        print(f"grid size = {l_value}")
        frame_data = analyze_square_frame(
                master_nodes,
                master_neighbors,
                L=l_value,
                boundary_thickness=args.bt
            )

        if frame_data['node_count'] == 0:
            print("  No nodes found in this frame. Skipping.")
            continue
        
        print(f"number of subnodes in the frame: {(frame_data['node_count'])}")
        print(f"number of top set: {len(frame_data['top_boundary_nodes'])}")
        print(f"number of bottom set: {len(frame_data['bottom_boundary_nodes'])}")
        print(f"number of left set: {len(frame_data['left_boundary_nodes'])}")
        print(f"number of right set: {len(frame_data['right_boundary_nodes'])}")
        
        print("="*60)
        
            
        MC_SIMU_I = percolationStatsI(nodes=frame_data['sub_graph_nodes'],
                                   neighbours=frame_data['sub_graph_neighbors'],
                                   top_set=frame_data['top_boundary_nodes'],
                                   bottom_set=frame_data['bottom_boundary_nodes'],
                                   left_set= frame_data['left_boundary_nodes'],
                                   right_set=frame_data['right_boundary_nodes'],
                                   trials=trials)
        MC_SIMU_I.report()
        
        meansI.append(MC_SIMU_I.trials_mean())
        stdsI.append(MC_SIMU_I.trials_std())
        
        MC_SIMU_U = percolationStatsU(nodes=frame_data['sub_graph_nodes'],
                                   neighbours=frame_data['sub_graph_neighbors'],
                                   top_set=frame_data['top_boundary_nodes'],
                                   bottom_set=frame_data['bottom_boundary_nodes'],
                                   left_set= frame_data['left_boundary_nodes'],
                                   right_set=frame_data['right_boundary_nodes'],
                                   trials=trials)
        MC_SIMU_U.report()
        meansU.append(MC_SIMU_U.trials_mean())
        stdsU.append(MC_SIMU_U.trials_std())
        
    
    
                
        
    print("\n")
    print("="*60)
    print("Generating visualization of the full patch and analysis frames...")
    
    print("\nGenerating visualization...")
    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    to_screen = [1, 0, 0, 0, 1, 0]
    
    try:
        patch.draw(to_screen, level=args.r + 1)
    except Exception as e:
        print(f"An error occurred during drawing: {e}")

    # --- MODIFIED: Draw rectangles at the new origin ---
    for l_value in l_values:
        rect = plt.Rectangle(
            (80 - l_value/2, 0 - l_value/2), # <-- Use new origin
            l_value, l_value, 
            fill=False, 
            edgecolor='red', 
            linewidth=1.5,
            linestyle='--'
        )
        ax.add_patch(rect)

    ax.set_aspect('equal', 'box')
    ax.axis('off')
    plt.title(f"Hat Tiling (Level {args.r}) with Frames Centered at ({80}, {0})", fontsize=16, pad=20)
    plt.show()
    
    print("\n")
    print("="*60)
    print("plot the percolation stats...")
    plot_percolation_stats_IU(l_values, meansI, stdsI, meansU, stdsU)
    
    print("\n")
    print("="*60)
    print("plot the extrapolates...")
    plot_extrapolation_IU(l_values, meansI, meansU)