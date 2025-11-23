# Reproduce Graph Results

To reproduce the graph results in the report:

> **Note:** These commands could take up to 5-10 minutes to run.

### Hat Tiling
```bash
python hat_tile_percolation_final.py --r 4 --t 200 --Lmin 30 --Lmax 150 --Lstep 20
```

### Penrose Rhombus Tiling
```bash
python penrose_rhombus_percolation_final.py -s 10 --Lmin 20 --Lmax 120 --Lstep 10 -t 1000
```

# Percolation Simulation on Aperiodic and Periodic Tilings

This project investigates percolation thresholds on various tiling structures, including the aperiodic Penrose Rhombus and Hat tilings, as well as standard periodic Square and Triangular lattices.

## Project Structure

The project consists of several Python scripts to generate tilings and perform Monte Carlo simulations for percolation analysis.

### Core Scripts

- **`hat_tile_percolation_final.py`**: Simulation on the "Hat" aperiodic monotile.
- **`penrose_rhombus_percolation_final.py`**: Simulation on Penrose Rhombus tilings.
- **`square_percolation_final.py`**: Simulation on the standard Square lattice.
- **`bond_percolation.py`**: Simulation of bond percolation on Square and Triangular lattices.
- **`visualize_penrose_rhombus.py`**: Helper module for generating and visualizing Penrose tilings.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`
- `scipy`
- `numba` (for acceleration)

Install dependencies via:
```bash
pip install numpy matplotlib scipy numba
```

## Usage

### 1. Hat Tiling Percolation
Run the simulation for the Hat tiling:
```bash
python hat_tile_percolation_final.py -r 4 -t 100
```
**Arguments:**
- `-r`: Recursion level (controls tiling size).
- `-t`: Number of Monte Carlo trials.

### 2. Penrose Rhombus Percolation
Analyze percolation on Penrose Rhombus tilings:
```bash
python penrose_rhombus_percolation_final.py -s 5 -t 100 --Lmin 20 --Lmax 100 --Lstep 20
```
**Arguments:**
- `-s`: Number of subdivisions (iterations) for tiling generation.
- `-t`: Number of trials.
- `--Lmin`, `--Lmax`, `--Lstep`: Range of frame sizes (L) to analyze.
- `--bt`: Boundary thickness for edge detection (default: 1.0).

### 3. Square Grid Percolation
Run site percolation analysis on a square grid:
```bash
python square_percolation_final.py --Lmin 20 --Lmax 100 --Lstep 20 --t 500
```
**Arguments:**
- `--Lmin`, `--Lmax`, `--Lstep`: Range of grid sizes (N).
- `--t`: Number of trials.

### 4. Bond Percolation
Simulate bond percolation on square and triangular lattices:
```bash
python bond_percolation.py
```
This script runs a comparison between Hat, Square, and Triangular tilings and produces summary plots.

## Visualizations

The scripts automatically generate and display plots using `matplotlib`, showing:
- Tiling structures
- Percolation probability curves
- Finite-size scaling analysis (extrapolation to infinite size)
