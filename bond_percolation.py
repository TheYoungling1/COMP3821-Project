import math
import numpy as np
from numba import njit, cuda, config
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

# Enable Numba disk caching for faster subsequent runs
config.CACHE_DIR = '.numba_cache'

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
    estimated_nodes = min(1000 * (4 ** level), 100000000)
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

# CPU using optimized Numba
@njit(cache=True)
def find_cpu(parent, x):
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
    parent = np.arange(N+2, dtype=np.int32)
    rank = np.zeros(N+2, dtype=np.int32)
    
    Lvirt, Rvirt = N, N+1
    sites = np.arange(N)
    np.random.shuffle(sites)
    
    for i in left_set:
        union_ranked_cpu(parent, rank, i, Lvirt)
    for i in right_set:
        union_ranked_cpu(parent, rank, i, Rvirt)
    
    percolates = np.zeros(N, dtype=np.float32)
    open_flags = np.zeros(N, dtype=np.uint8)
    
    for k in range(N):
        site = sites[k]
        open_flags[site] = 1
        
        start, end = neighbor_starts[site], neighbor_starts[site+1]
        for j_idx in range(start, end):
            nb = neighbors_arr[j_idx]
            if open_flags[nb]:
                union_ranked_cpu(parent, rank, site, nb)
        
        if find_cpu(parent, Lvirt) == find_cpu(parent, Rvirt):
            percolates[k:] = 1.0
            break
    
    return percolates

def site_crossing_curve_nz_cpu(neighbors_arr, neighbor_starts, left_set, right_set, N, trials=100):
    """CPU Newman-Ziff algorithm."""
    total = np.zeros(N, dtype=np.float32)
    for t in range(trials):
        if (t+1) % 50 == 0:
            print(f"  Progress: {t+1}/{trials} trials")
        total += newman_ziff_trial_cpu(neighbors_arr, neighbor_starts, N, left_set, right_set)
    probs = total / trials
    p_list = np.arange(1, N+1) / N
    return p_list, probs

def smooth_curve(p_grid, probs, window_size=50):
    """Apply moving average smoothing to the curve."""
    if window_size <= 1:
        return p_grid, probs
    
    probs = np.asarray(probs)
    p_grid = np.asarray(p_grid)
    
    # Pad the array to handle edges
    pad_size = window_size // 2
    probs_padded = np.pad(probs, pad_size, mode='edge')
    
    # Apply moving average
    smoothed = np.convolve(probs_padded, np.ones(window_size)/window_size, mode='valid')
    
    # Trim to match original length
    if len(smoothed) > len(probs):
        start = (len(smoothed) - len(probs)) // 2
        smoothed = smoothed[start:start+len(probs)]
    
    return p_grid, smoothed

def estimate_pc_half_height(p_grid, probs):
    probs = np.asarray(probs)
    p_grid = np.asarray(p_grid)
    mono = np.maximum.accumulate(probs)
    target = np.clip(0.5, mono.min(), mono.max())
    return float(np.interp(target, mono, p_grid))

def estimate_pc_max_slope(p_grid, probs):
    probs = np.asarray(probs)
    p_grid = np.asarray(p_grid)
    dp = p_grid[2:] - p_grid[:-2]
    dP = probs[2:] - probs[:-2]
    slope = dP / dp
    k = np.argmax(slope)
    pc = (p_grid[k] + p_grid[k+2]) / 2.0
    return float(pc), float(slope[k])

def build_square_grid(n):
    """Build n x n square lattice."""
    nodes = []
    for i in range(n):
        for j in range(n):
            nodes.append([i, j])
    nodes = np.array(nodes)
    
    neighbors = []
    for i in range(n):
        for j in range(n):
            idx = i*n + j
            nbrs = []
            if i > 0: nbrs.append((i-1)*n + j)
            if i < n-1: nbrs.append((i+1)*n + j)
            if j > 0: nbrs.append(i*n + (j-1))
            if j < n-1: nbrs.append(i*n + (j+1))
            neighbors.append(np.array(nbrs, dtype=np.int32))
    return nodes, neighbors

def build_triangular_grid(n):
    """Build n x n triangular lattice."""
    nodes = []
    for j in range(n):
        for i in range(n):
            x = i + 0.5*(j % 2)
            y = (math.sqrt(3)/2) * j
            nodes.append([x, y])
    nodes = np.array(nodes)
    
    neighbors = []
    for j in range(n):
        for i in range(n):
            idx = j*n + i
            nbrs = []
            if i > 0: nbrs.append(j*n + (i-1))
            if i < n-1: nbrs.append(j*n + (i+1))
            if j > 0:
                nbrs.append((j-1)*n + i)
                if (j % 2 == 0 and i < n-1):
                    nbrs.append((j-1)*n + (i+1))
                if (j % 2 == 1 and i > 0):
                    nbrs.append((j-1)*n + (i-1))
            if j < n-1:
                nbrs.append((j+1)*n + i)
                if (j % 2 == 0 and i < n-1):
                    nbrs.append((j+1)*n + (i+1))
                if (j % 2 == 1 and i > 0):
                    nbrs.append((j+1)*n + (i-1))
            neighbors.append(np.array(nbrs, dtype=np.int32))
    return nodes, neighbors



# Bond Percolation
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        # union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

class BondPercolation:
    def __init__(self, N, lattice_type="square", uf_class=None):
        self.N = N
        self.lattice_type = lattice_type
        self.uf_class = uf_class  # optional existing Union-Find
        self.edges = self._generate_edges()
        self.uf = self.uf_class(N * N) if uf_class else None
        self.open_edges = set()

    def _index(self, i, j):
        """Convert 2D coordinate to 1D index."""
        return i * self.N + j

    def _generate_edges(self):
        """Generate all possible bonds (edges) for the lattice."""
        edges = []
        for i in range(self.N):
            for j in range(self.N):
                # Square lattice edges
                if self.lattice_type == "square":
                    if i + 1 < self.N:  # vertical
                        edges.append(((i, j), (i + 1, j)))
                    if j + 1 < self.N:  # horizontal
                        edges.append(((i, j), (i, j + 1)))

                # Triangular lattice adds diagonal edges
                elif self.lattice_type == "triangle":
                    if i + 1 < self.N:
                        edges.append(((i, j), (i + 1, j)))      # vertical
                        if j + 1 < self.N:
                            edges.append(((i, j), (i + 1, j + 1)))  # diagonal
                    if j + 1 < self.N:
                        edges.append(((i, j), (i, j + 1)))      # horizontal
        return edges

    def open_bonds(self, p):
        """Randomly open bonds with probability p."""
        self.open_edges = [e for e in self.edges if random.random() < p]
        if self.uf:
            self.uf = self.uf_class(self.N * self.N)
            for (a, b) in self.open_edges:
                a_idx = self._index(*a)
                b_idx = self._index(*b)
                self.uf.union(a_idx, b_idx)

    def percolates(self):
        """Check if there's a spanning cluster from top to bottom."""
        if not self.uf:
            raise ValueError("Union-Find not initialized")

        top = [self._index(0, j) for j in range(self.N)]
        bottom = [self._index(self.N - 1, j) for j in range(self.N)]

        for a in top:
            for b in bottom:
                if self.uf.connected(a, b): 
                    return True
        return False

def simulate_bond_percolation(N=100, lattice_type="square", trials=50, p_step=0.02):
    """
    Run bond percolation simulations for a given lattice and plot percolation probability vs p.
    Returns the estimated p_c (where crossing prob = 0.5).
    """
    p_values = np.arange(0.0, 1.01, p_step)
    percolation_probs = []

    for p in p_values:
        count = 0
        for _ in range(trials):
            bp = BondPercolation(N, lattice_type=lattice_type, uf_class=UnionFind)
            bp.open_bonds(p)
            if bp.percolates():
                count += 1
        prob = count / trials
        percolation_probs.append(prob)

        # print progress at 0.1 increments (0.0, 0.1, 0.2, ...)
        if abs((p * 10) - round(p * 10)) < 1e-9:
            print(f"p = {p:.2f}, percolation probability = {prob:.3f}")

    p_values = np.asarray(p_values)
    percolation_probs = np.asarray(percolation_probs, dtype=float)

    # enforce monotonicity (crossing prob should be non-decreasing with p)
    mono_probs = np.maximum.accumulate(percolation_probs)

    # clip the target in case the curve never reaches 0.5
    target = np.clip(0.5, mono_probs.min(), mono_probs.max())

    # interpolate to find p where crossing probability = 0.5
    p_c = float(np.interp(target, mono_probs, p_values))

    # For reporting: compute interpolated probability at p_c (should equal target)
    prob_at_pc = float(np.interp(p_c, p_values, percolation_probs))

    print("\nEstimated p_c (empirical, P=0.5):", f"{p_c:.5f}")
    print("Interpolated crossing probability at p_c:", f"{prob_at_pc:.5f}")

    # Plot the curve and vertical line at the estimated p_c
    plt.figure(figsize=(8,6))
    plt.plot(p_values, percolation_probs, '-', label=f"{lattice_type} lattice (N={N}, trials={trials})")
    plt.plot(p_values, mono_probs, ':', linewidth=1, label='monotone(enforced) curve')
    plt.axvline(p_c, color='r', linestyle='--', label=f'Estimated p_c = {p_c:.4f}')
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='P = 0.5')

    plt.xlabel("Bond Open Probability (p)")
    plt.ylabel("Percolation Probability (Top-Bottom Connection)")
    plt.title(f"Bond Percolation on {lattice_type.capitalize()} Lattice")
    plt.legend(loc='best')
    plt.grid(True)

    plt.ylim(-0.02, 1.02)
    plt.show()

    return p_c


# Main execution
if __name__ == "__main__":

    print("\nBuilding tiles...")
    tiles0 = [H_init(), T_init(), P_init(), F_init()]
    patch1 = constructPatch(*tiles0)
    tiles1 = constructMetatiles(patch1)
    patch2 = constructPatch(*tiles1)
    tiles2 = constructMetatiles(patch2)
    patch3 = constructPatch(*tiles2)
    tiles3 = constructMetatiles(patch3)
    patch4 = constructPatch(*tiles3)
    tiles4 = constructMetatiles(patch4)
    patch5 = constructPatch(*tiles4)
    tiles5 = constructMetatiles(patch5)
    patch6 = constructPatch(*tiles5)

    print("Building neighbor graph...")
    unique_nodes, neighbors = build_neighbor_graph_fast(patch4, level=5)
    
    print("Converting to CSR format...")
    neighbors_arr, neighbor_starts = neighbors_to_csr(neighbors)
    
    all_x = unique_nodes[:, 0]
    xmin, xmax = all_x.min(), all_x.max()
    tol = 1e-5
    left_set = np.where(all_x <= xmin + tol)[0].astype(np.int32)
    right_set = np.where(all_x >= xmax - tol)[0].astype(np.int32)
    N = len(unique_nodes)
    
    print(f"Graph has {N} nodes")
    print(f"Left boundary: {len(left_set)} nodes, Right boundary: {len(right_set)} nodes")
    
    # HAT TILING
    print("\n" + "="*60)
    print("HAT TILING PERCOLATION")
    print("="*60)
    
    t0 = time.time()
    p_list, probs = site_crossing_curve_nz_cpu(neighbors_arr, neighbor_starts, 
                                                    left_set, right_set, N, trials=100)
    mask = p_list >= 0.8
    p_list = p_list[mask]
    probs = probs[mask]
    t1 = time.time()
    print(f"✓ Completed in {t1-t0:.2f}s")
    
    pc_half = estimate_pc_half_height(p_list, probs)
    pc_slope, max_slope = map(lambda x: float(np.float64(x)), estimate_pc_max_slope(p_list, probs))
    
    print(f"\nResults:")
    print(f"  p_c (P=0.5): {pc_half:.4f}")
    print(f"  p_c (max slope): {pc_slope:.10f}, max slope: {max_slope:.10f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_list, probs, 'b-', linewidth=2, label='Crossing probability')
    plt.axvline(pc_half, color='r', linestyle='--', label=f'p_c (P=0.5) = {pc_half:.4f}')
    plt.axvline(pc_slope, color='g', linestyle='--', label=f'p_c (max slope) = {pc_slope:.10f}')
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel("p (site open probability)", fontsize=12)
    plt.ylabel("Left-right crossing probability", fontsize=12)
    plt.title("Site Percolation on Hat Tiling", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()
    
    # Draw tiling
    print("\nDrawing tiling...")
    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    to_screen = [1, 0, 0, 0, 1, 0]
    patch2.draw(to_screen, level=4)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.title("Hat Tile Aperiodic Tiling", fontsize=16, pad=20)
    plt.show()

    # SQUARE GRID
    print("\n" + "="*60)
    print("SQUARE LATTICE PERCOLATION")
    print("="*60)
    n = 200
    print(f"Building {n}×{n} square grid...")
    square_nodes, square_neighbors = build_square_grid(n)
    neighbors_arr_sq, neighbor_starts_sq = neighbors_to_csr(square_neighbors)
    
    left_set_sq = np.arange(0, n, dtype=np.int32)
    right_set_sq = np.arange(n*(n-1), n*n, dtype=np.int32)
    N_sq = n*n
    
    print(f"Square grid: {N_sq} nodes")
    print("Running CPU simulation...")
    t0 = time.time()
    p_list_sq, probs_sq = site_crossing_curve_nz_cpu(neighbors_arr_sq, neighbor_starts_sq,
                                                        left_set_sq, right_set_sq, N_sq, trials=100)
    t1 = time.time()
    print(f"✓ Completed in {t1-t0:.2f}s")
    
    pc_half_sq = estimate_pc_half_height(p_list_sq, probs_sq)
    pc_slope_sq, max_slope_sq = estimate_pc_max_slope(p_list_sq, probs_sq)
    
    print(f"\nResults:")
    print(f"  p_c (P=0.5): {pc_half_sq:.4f}")
    print(f"  p_c (max slope): {pc_slope_sq:.4f}")
    print(f"  Theory: p_c = 0.5927")
    
    plt.figure(figsize=(10,6))
    plt.plot(p_list_sq, probs_sq, 'b-', linewidth=2, label='Square grid crossing probability')
    plt.axvline(pc_half_sq, color='r', linestyle='--', label=f'p_c (P=0.5) = {pc_half_sq:.4f}')
    plt.axvline(pc_slope_sq, color='g', linestyle='--', label=f'p_c (max slope) = {pc_slope_sq:.4f}')
    plt.axvline(0.5927, color='purple', linestyle=':', label='Theory: 0.5927', alpha=0.7)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel("p (site open probability)", fontsize=12)
    plt.ylabel("Left-right crossing probability", fontsize=12)
    plt.title("Site Percolation on Square Grid", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()
    
    # TRIANGULAR GRID
    print("\n" + "="*60)
    print("TRIANGULAR LATTICE PERCOLATION")
    print("="*60)
    n_tri = 200
    print(f"Building {n_tri}×{n_tri} triangular grid...")
    tri_nodes, tri_neighbors = build_triangular_grid(n_tri)
    neighbors_arr_tri, neighbor_starts_tri = neighbors_to_csr(tri_neighbors)
    
    all_x_tri = tri_nodes[:,0]
    tol = 1e-5
    left_set_tri = np.where(all_x_tri <= all_x_tri.min() + tol)[0].astype(np.int32)
    right_set_tri = np.where(all_x_tri >= all_x_tri.max() - tol)[0].astype(np.int32)
    N_tri = len(tri_nodes)
    
    print(f"Triangular grid: {N_tri} nodes")

    print("Running CPU simulation...")
    t0 = time.time()
    p_list_tri, probs_tri = site_crossing_curve_nz_cpu(neighbors_arr_tri, neighbor_starts_tri,
                                                            left_set_tri, right_set_tri, N_tri, trials=5000)
    t1 = time.time()
    print(f"✓ Completed in {t1-t0:.2f}s")
    
    pc_half_tri = estimate_pc_half_height(p_list_tri, probs_tri)
    pc_slope_tri, max_slope_tri = estimate_pc_max_slope(p_list_tri, probs_tri)
    
    print(f"\nResults:")
    print(f"  p_c (P=0.5): {pc_half_tri:.4f}")
    print(f"  p_c (max slope): {pc_slope_tri:.4f}")
    print(f"  Theory: p_c = 0.5000")
    
    plt.figure(figsize=(10,6))
    plt.plot(p_list_tri, probs_tri, 'b-', linewidth=2, label='Triangular grid crossing probability')
    plt.axvline(pc_half_tri, color='r', linestyle='--', label=f'p_c (P=0.5) = {pc_half_tri:.4f}')
    plt.axvline(pc_slope_tri, color='g', linestyle='--', label=f'p_c (max slope) = {pc_slope_tri:.4f}')
    plt.axvline(0.5, color='purple', linestyle=':', label='Theory: 0.5000', alpha=0.7)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel("p (site open probability)", fontsize=12)
    plt.ylabel("Left-right crossing probability", fontsize=12)
    plt.title("Site Percolation on Triangular Lattice", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

    # SUMMARY
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"Hat Tiling:          p_c = {pc_half:.4f}")
    print(f"Square Lattice:      p_c = {pc_half_sq:.4f} (Theory: 0.5927)")
    print(f"Triangular Lattice:  p_c = {pc_half_tri:.4f} (Theory: 0.5000)")
    print("="*60)
    
    # Combined plot
    plt.figure(figsize=(12, 7))
    plt.plot(p_list, probs, 'b-', linewidth=2, label=f'Hat Tiling (p_c={pc_half:.3f})', alpha=0.8)
    plt.plot(p_list_sq, probs_sq, 'r-', linewidth=2, label=f'Square (p_c={pc_half_sq:.3f})', alpha=0.8)
    plt.plot(p_list_tri, probs_tri, 'g-', linewidth=2, label=f'Triangular (p_c={pc_half_tri:.3f})', alpha=0.8)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel("p (site open probability)", fontsize=13)
    plt.ylabel("Left-right crossing probability", fontsize=13)
    plt.title("Percolation Comparison: Hat Tiling vs Regular Lattices", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

    simulate_bond_percolation(N=100, lattice_type="square", trials=30, p_step=0.01)
    simulate_bond_percolation(N=100, lattice_type="triangle", trials=30, p_step=0.01)