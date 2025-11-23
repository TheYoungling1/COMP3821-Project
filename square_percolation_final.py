import numpy as np
import random
import math
import matplotlib.pyplot as plt
import argparse
from scipy.stats import linregress

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

class percolationI:
    # create a n by n grid with all sites blocked
    def __init__(self, n: int):
        if n <= 0: raise ValueError("n must be a positive integer")
        
        self.gridSize = n
        self.gridSquare = n * n
        self.grid = np.array([[0 for _ in range (n)] for _ in range (n)])
        
        self.wqfGrid_LR = WeightedQuickUnionUF(self.gridSquare + 2)
        self.wqfGrid_TB = WeightedQuickUnionUF(self.gridSquare + 2)
        
        self.virtualTop = self.gridSquare
        self.virtualBottom = self.gridSquare + 1
        
        self.virtualLeft = self.gridSquare
        self.virtualRight = self.gridSquare + 1
        
        self.openSite = 0
    
    # open the site[i,j] if it's not open yet
    def open_site(self, row: int, col: int):
        self.validState(row, col)
        
        shiftRow = row - 1
        shiftCol = col - 1
        flatIndex = self.flattenGrid(row, col) - 1
        
        if(self.isOpen(row, col)):
            return
        
        self.grid[shiftRow][shiftCol] = 1
        self.openSite += 1
        
        # connect to neighbours
        
        ## top row
        if (row == 1):
            self.wqfGrid_TB.union(self.virtualTop, flatIndex)
            
        
        ## bottom row
        if (row == self.gridSize):
            self.wqfGrid_TB.union(self.virtualBottom, flatIndex)
            
        ## leftmost
        # if (col == 1 and row != 1 and row != self.gridSize):
        #     self.wqfGrid_LR.union(self.virtualLeft, flatIndex)
        
        if (col == 1):
            self.wqfGrid_LR.union(self.virtualLeft, flatIndex)
        
        ## rightmost
        # if (col == self.gridSize and row != 1 and row != self.gridSize):
        #     self.wqfGrid_LR.union(self.virtualRight, flatIndex)
        
        if (col == self.gridSize):
            self.wqfGrid_LR.union(self.virtualRight, flatIndex)
        
        ## left
        if (self.isOnGrid(row, col-1)) and (self.isOpen(row, col-1)):
            self.wqfGrid_TB.union(flatIndex, self.flattenGrid(row, col-1) - 1)
            self.wqfGrid_LR.union(flatIndex, self.flattenGrid(row, col-1) - 1)
            
        
        
        ## right
        if (self.isOnGrid(row, col+1)) and (self.isOpen(row, col+1)):
            self.wqfGrid_TB.union(flatIndex, self.flattenGrid(row, col+1) - 1)
            self.wqfGrid_LR.union(flatIndex, self.flattenGrid(row, col+1) - 1)
            
            
        ## up
        if (self.isOnGrid(row - 1, col)) and (self.isOpen(row - 1, col)):
            self.wqfGrid_TB.union(flatIndex, self.flattenGrid(row - 1, col) - 1)
            self.wqfGrid_LR.union(flatIndex, self.flattenGrid(row - 1, col) - 1)
            
           
        ## down
        if (self.isOnGrid(row + 1, col)) and (self.isOpen(row + 1, col)):
            self.wqfGrid_TB.union(flatIndex, self.flattenGrid(row + 1, col) - 1)
            self.wqfGrid_LR.union(flatIndex, self.flattenGrid(row + 1, col) - 1)

    # is site[i,j] open?
    def isOpen(self, row: int, col: int) -> bool:
        self.validState(row, col)
        return self.grid[row - 1][col - 1]
    
    def percolates(self, ) -> bool:
        return self.wqfGrid_TB.connected(self.virtualTop, self.virtualBottom) and self.wqfGrid_LR.connected(self.virtualLeft, self.virtualRight)
            
    
    def numberOfOpenSites(self,) -> int:
        return self.openSite
    
    def validState(self, row: int, col: int):
        if not self.isOnGrid(row, col):
            raise IndexError("index is out of bounds")
        
    def flattenGrid(self, row: int, col: int) -> int:
        return self.gridSize * (row - 1) + col
        
    
    def isOnGrid(self, row: int, col: int) -> bool:
        shiftRow = row - 1
        shiftCol = col - 1
        
        return (shiftRow >= 0 and shiftCol >= 0 and shiftRow < self.gridSize and shiftCol < self.gridSize)
    
class percolationStatsI:
    def __init__(self, n: int, trials: int):
                
        if (n <= 0 or trials <= 0):
            raise ValueError("grid size n and trials count must be postive integer")

        self.trialCount = trials
        self.gridSize = n
        self.trialResults = []
        
        for i in range(self.trialCount):
            simulator = percolationI(self.gridSize)
            while (not simulator.percolates()):
                row = random.randint(1, self.gridSize)
                col = random.randint(1, self.gridSize)
                simulator.open_site(row, col)
                
            openSites = simulator.openSite
            result = openSites / (self.gridSize * self.gridSize)
            self.trialResults.append(result)
            
            
    def trials_mean(self, ):
        return np.mean(self.trialResults)

    def trials_std(self, ):
        return np.std(self.trialResults)
    
    def trails_confidence_interval(self, ):
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


class percolationU:
    # create a n by n grid with all sites blocked
    def __init__(self, n: int):
        if n <= 0: raise ValueError("n must be a positive integer")
        
        self.gridSize = n
        self.gridSquare = n * n
        self.grid = np.array([[0 for _ in range (n)] for _ in range (n)])
        
        self.wqfGrid_LR = WeightedQuickUnionUF(self.gridSquare + 2)
        self.wqfGrid_TB = WeightedQuickUnionUF(self.gridSquare + 2)
        
        self.virtualTop = self.gridSquare
        self.virtualBottom = self.gridSquare + 1
        
        self.virtualLeft = self.gridSquare
        self.virtualRight = self.gridSquare + 1
        
        self.openSite = 0
    
    # open the site[i,j] if it's not open yet
    def open_site(self, row: int, col: int):
        self.validState(row, col)
        
        shiftRow = row - 1
        shiftCol = col - 1
        flatIndex = self.flattenGrid(row, col) - 1
        
        if(self.isOpen(row, col)):
            return
        
        self.grid[shiftRow][shiftCol] = 1
        self.openSite += 1
        
        # connect to neighbours
        
        ## top row
        if (row == 1):
            self.wqfGrid_TB.union(self.virtualTop, flatIndex)
            
        
        ## bottom row
        if (row == self.gridSize):
            self.wqfGrid_TB.union(self.virtualBottom, flatIndex)
            
        ## leftmost
        # if (col == 1 and row != 1 and row != self.gridSize):
        #     self.wqfGrid_LR.union(self.virtualLeft, flatIndex)
        
        if (col == 1):
            self.wqfGrid_LR.union(self.virtualLeft, flatIndex)
        
        ## rightmost
        # if (col == self.gridSize and row != 1 and row != self.gridSize):
        #     self.wqfGrid_LR.union(self.virtualRight, flatIndex)
            
        if (col == self.gridSize):
            self.wqfGrid_LR.union(self.virtualRight, flatIndex)
        
        ## left
        if (self.isOnGrid(row, col-1)) and (self.isOpen(row, col-1)):
            self.wqfGrid_TB.union(flatIndex, self.flattenGrid(row, col-1) - 1)
            self.wqfGrid_LR.union(flatIndex, self.flattenGrid(row, col-1) - 1)
            
        
        
        ## right
        if (self.isOnGrid(row, col+1)) and (self.isOpen(row, col+1)):
            self.wqfGrid_TB.union(flatIndex, self.flattenGrid(row, col+1) - 1)
            self.wqfGrid_LR.union(flatIndex, self.flattenGrid(row, col+1) - 1)
            
            
        ## up
        if (self.isOnGrid(row - 1, col)) and (self.isOpen(row - 1, col)):
            self.wqfGrid_TB.union(flatIndex, self.flattenGrid(row - 1, col) - 1)
            self.wqfGrid_LR.union(flatIndex, self.flattenGrid(row - 1, col) - 1)
            
           
        ## down
        if (self.isOnGrid(row + 1, col)) and (self.isOpen(row + 1, col)):
            self.wqfGrid_TB.union(flatIndex, self.flattenGrid(row + 1, col) - 1)
            self.wqfGrid_LR.union(flatIndex, self.flattenGrid(row + 1, col) - 1)
            
     
        
    
    # is site[i,j] open?
    def isOpen(self, row: int, col: int) -> bool:
        self.validState(row, col)
        return self.grid[row - 1][col - 1]
    
    def percolates(self, ) -> bool:
        return self.wqfGrid_TB.connected(self.virtualTop, self.virtualBottom) or self.wqfGrid_LR.connected(self.virtualLeft, self.virtualRight)
            
    
    def numberOfOpenSites(self,) -> int:
        return self.openSite
    
    def validState(self, row: int, col: int):
        if not self.isOnGrid(row, col):
            raise IndexError("index is out of bounds")
        
    def flattenGrid(self, row: int, col: int) -> int:
        return self.gridSize * (row - 1) + col
        
    
    def isOnGrid(self, row: int, col: int) -> bool:
        shiftRow = row - 1
        shiftCol = col - 1
        
        return (shiftRow >= 0 and shiftCol >= 0 and shiftRow < self.gridSize and shiftCol < self.gridSize)
    
class percolationStatsU:
    def __init__(self, n: int, trials: int):
                
        if (n <= 0 or trials <= 0):
            raise ValueError("grid size n and trials count must be postive integer")

        self.trialCount = trials
        self.gridSize = n
        self.trialResults = []
        
        for i in range(self.trialCount):
            simulator = percolationU(self.gridSize)
            while (not simulator.percolates()):
                row = random.randint(1, self.gridSize)
                col = random.randint(1, self.gridSize)
                simulator.open_site(row, col)
                
            openSites = simulator.openSite
            result = openSites / (self.gridSize * self.gridSize)
            self.trialResults.append(result)
            
            
    def trials_mean(self, ):
        return np.mean(self.trialResults)

    def trials_std(self, ):
        return np.std(self.trialResults)
    
    def trails_confidence_interval(self, ):
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
        


def plot_percolation_stats_RD(L_values, meansR, stdsR, meansD, stdsD, confidence_level=1.96):
    """
    Generates an error bar plot comparing the mean critical probabilities 
    for Rightward (R) and Downward (D) percolation vs L.
    """
    
    plt.figure(figsize=(10, 6))
    
    # --- Plot 1: Rightward (R) Data ---
    plt.errorbar(
        L_values, 
        meansR, 
        yerr=stdsR, 
        fmt='o-',               # Circle markers, connected line
        color='blue', 
        ecolor='blue',          # Error bars color
        capsize=5,              
        label='Mean $p_c^R \pm \sigma_R$'
    )
    
    # --- Plot 2: Downward (D) Data ---
    # Shift L values slightly for the second plot to prevent marker overlap
    L_shifted = L_values * 1.01 
    plt.errorbar(
        L_shifted, 
        meansD, 
        yerr=stdsD, 
        fmt='s-',               # Square markers, connected line
        color='red', 
        ecolor='red',           # Error bars color
        capsize=5,              
        label='Mean $p_c^D \pm \sigma_D$'
    )
    
    # Add labels and title
    plt.xlabel('Linear System Size ($L$)', fontsize=14)
    plt.ylabel('Mean Critical Probability ($\\bar{p}_c$)', fontsize=14)
    plt.title('Mean $p_c$ vs. System Size ($L$) for R and D Criteria', fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    plt.show()

# --------------------------------------------------------------------------------------

### 2. Extrapolation Plot (`plot_extrapolation_RD`)
def plot_extrapolation_RD(L_values, meansR, meansD, exponent=-3/4):
    """
    Plots mean critical probability vs L^(exponent) for R, D, and A=(R+D)/2,
    extrapolates lines, and estimates pc(infinity) for each.
    """
    # Ensure arrays
    L_values = np.asarray(L_values, dtype=float)
    meansR = np.asarray(meansR, dtype=float)
    meansD = np.asarray(meansD, dtype=float)
    meansA = 0.5 * (meansR + meansD)

    # 1) Scaling variable
    X_scaling = L_values ** exponent

    plt.figure(figsize=(10, 6))

    results = {}

    # Define plotting configs
    data_sets = {
        'R': {'means': meansR, 'color': 'blue',  'marker': 'o', 'label': 'Rightward ($p_c^R$)'},
        'D': {'means': meansD, 'color': 'red',   'marker': 's', 'label': 'Downward ($p_c^D$)'},
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
    plt.title('Finite-Size Scaling Extrapolation for R, D, and Average (A)', fontsize=16)

    plt.xlim(X_plot_min - 0.05 * X_plot_max, X_plot_max)
    all_y = np.concatenate([meansR, meansD, meansA])
    plt.ylim(0.95 * float(np.min(all_y)), 1.05 * float(np.max(all_y)))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')

    # 4) Print summary
    print(f"\n--- Extrapolation Results (exponent {exponent:.2f}) ---")
    for key, res in results.items():
        name = {'R':'Rightward', 'D':'Downward', 'A':'Average'}[key]
        print(f"{name}: pc(infinity) = {res['pc_inf']:.6f}, R^2 = {res['R2']:.4f}")
    print("-------------------------------------------------------")

    plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run a Monte Carlo simulation for 2D percolation."
    )
    
    parser.add_argument(
        '--Lmin', 
        type=int, 
        default=50, 
        help="Minimum size of the square grid (N_min x N_min)."
    )
    
    parser.add_argument(
        '--Lmax', 
        type=int, 
        default=200, 
        help="Maximum size of the square grid (N_max x N_max)."
    )
    
    parser.add_argument(
        '--Lstep', 
        type=int, 
        default=50, 
        help="Step size for increasing the grid size N."
    )
    
    
    parser.add_argument(
        '--t', 
        type=int, 
        default=500, 
        help="The number of Monte Carlo trials to perform."
    )
    
    args = parser.parse_args()
    
 
    L_min = args.Lmin
    L_max = args.Lmax
    L_step = args.Lstep
    trials = args.t
    
    print("Starting Monte Carlo Percolation Analysis...")
    print(f"System sizes (N): {L_min} to {L_max}, step {L_step}")
    print(f"Trials per size: {trials}")
    

    L_values = []
    meansI = []
    stdsI = []
    
    meansU = []
    stdsU = []

    for n_value in np.arange(L_min, L_max + 1, L_step):
        
        L_values.append(n_value)
        print(f"simulate n = {n_value}")
        
        MC_SIMU_I = percolationStatsI(
            n = n_value,
            trials = trials
        )
        
 
        MC_SIMU_I.report()
        meansI.append(MC_SIMU_I.trials_mean())
        stdsI.append(MC_SIMU_I.trials_std())
        
        MC_SIMU_U = percolationStatsU(
            n = n_value,
            trials = trials
        )
        
 
        MC_SIMU_U.report()
        meansU.append(MC_SIMU_U.trials_mean())
        stdsU.append(MC_SIMU_U.trials_std())
        

    # print(f"all means: {all_means}")
    # print(f"all std: {all_stds}")

    print("\n--- Simulation Complete ---")
    
    print("="*60)
    print("plotting...")
    plot_percolation_stats_RD(np.array(L_values), meansI, stdsI, meansU, stdsU)
    plot_extrapolation_RD(np.array(L_values), meansI, meansU)