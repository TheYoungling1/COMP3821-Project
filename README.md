# COMP3821-Group-Project

Install dependencies by 

pip install -r requirements.txt

## How to Run

Run Penrose rhombus tiling percolation analysis:

```bash
python penrose_rhombus_percolation.py -s NUM_OF_SUBDIVISIONS
```

**Parameters:**
- `NUM_OF_SUBDIVISIONS`: Number of recursive iterations used to generate Penrose tiling
  - Level 11: ~140,000 vertices
  - Level 12: ~375,000 vertices
  - Level 13: ~980,000 vertices
  - **Recommended:** Level 11-12 for balance between sample size and accuracy in simulation
  - **Warning:** Level 13 takes ~30 minutes to generate graph
