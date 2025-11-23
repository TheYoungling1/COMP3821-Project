import math
import cmath
import random

# Golden ratio
phi = (5 ** 0.5 + 1) / 2

class PenroseTriangle:
    """
    A class that represents individual Robinson's triangles, the "thick" triangle (36, 72, 72), and the "thin" triangle (36, 108, 36)
    """

    def __init__(self, shape, v1, v2, v3):
        """
        Intializes a triangle with its type "thick"/"thin" and their 3 corresponding vertices, 
        where the edge v1-v2 always represent the base edge of the triangle
        """   
        self.shape = shape
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def subdivide(self):
        """
        Subdivide triangles according to the rules in:
        https://www.chiark.greenend.org.uk/~sgtatham/quasiblog/aperiodic-tilings/
        """
        if self.shape == "thin":
            # Divide thin triangle into 1 thin + 1 thick
            p1 = self.v1 + (self.v2 - self.v1) / phi
            return [
                PenroseTriangle("thin", self.v3, p1, self.v2),
                PenroseTriangle("thick", p1, self.v3, self.v1)
            ]
        else:  # thick
            # Divide thick triangle into 1 thin + 2 thick
            p2 = self.v2 + (self.v1 - self.v2) / phi
            p3 = self.v2 + (self.v3 - self.v2) / phi
            return [
                PenroseTriangle("thick", p3, self.v3, self.v1),
                PenroseTriangle("thick", p2, p3, self.v2),
                PenroseTriangle("thin", p3, p2, self.v1)
            ]

    def get_all_vertices(self):
        """Return all 3 vertices of the triangle."""
        return [self.v1, self.v2, self.v3]

    def get_edges_for_percolation(self):
        """
        Return edges for percolation analysis.

        Removes the base edges in the triangles where they combine with one another, i.e. removing the edges of symmetry in the rhombus

        For thin triangles: returns only the two non-base edges (v1-v3 and v2-v3)
        For thick triangles: returns only the two non-base edges (v1-v3 and v2-v3)

        So the the percolation is done on rhombus shapes, not triangle shapes
        """
        
        return [
            (self.v1, self.v3),
            (self.v1, self.v2)
        ]

class PenroseTiling:
    """
    A class for subdividing the Robinson's triangles and form them into a valid tiling
    """

    def __init__(self, divisions=4, base=5, scale=200, config=None):
        """
        Initialize Penrose tiling generator.

        Args:
            divisions: Number of subdivision iterations
            base: Number of initial triangles (5 for standard sun pattern)
            scale: SVG scale factor
            config: Dictionary of rendering options
        """

        self.divisions = divisions
        self.base = base
        self.scale = scale

        # Default configuration
        self.config = {
            'width': '100%',
            'height': '100%',
            'margin': 1.05,
            'thin-colour': '#0035f3',
            'thick-colour': '#08f',
            'stroke-colour': '#fff',
            'stroke-width': 0.05,
            'tile-opacity': 0.8,
            'random-tile-colours': False,
        }
        if config:
            self.config.update(config)

        self.triangles = []

    def create_initial_tiles(self):
        """
        Creating the initial "star" pattern by using 10 thin triangles arranged in a 5-pointed star shape.
        Each pair of triangles forms a thin rhombus pointing outward to create the star points.
        This creates a star pattern instead of a regular decagon.
        """
        initial_scale = self.scale * 0.5
        triangles = []

        # For a 5-pointed star pattern, arrange triangles so pairs form rhombi pointing outward
        # Star points are at: 90°, 18°, -54°, -126°, 162° (π/2, π/10, -3π/10, -7π/10, 9π/10)
        # Each star point uses 2 triangles forming a rhombus
        
        for i in range(self.base * 2):
            # Which star point (0-4) and which triangle in pair (0 or 1)
            star_idx = i // 2
            is_first_triangle = (i % 2 == 0)
            
            # Star point direction: start at 90° (π/2), rotate by -72° (2π/5) for each point
            star_direction = math.pi / 2 - star_idx * 2 * math.pi / self.base
            
            # Thin triangle angle spread: 36° = π/5
            angle_spread = math.pi / self.base
            
            if is_first_triangle:
                # First triangle of pair: points toward star tip
                v2 = cmath.rect(initial_scale, star_direction - angle_spread)
                v3 = cmath.rect(initial_scale, star_direction + angle_spread)
            else:
                # Second triangle: mirrored to complete the rhombus pointing outward
                v2 = cmath.rect(initial_scale, star_direction + angle_spread)
                v3 = cmath.rect(initial_scale, star_direction - angle_spread)
                # Swap to create proper reflection
                v2, v3 = v3, v2

            triangles.append(PenroseTriangle("thin", 0, v2, v3))

        self.triangles = triangles

    def subdivide_all(self):
        """Perform all recursive subdivision iterations, stores results in an array"""
        for _ in range(self.divisions):
            new_triangles = []
            for rhombus in self.triangles:
                new_triangles.extend(rhombus.subdivide())
            self.triangles = new_triangles

    def make_tiling(self):
        """Generate the complete Penrose tiling."""
        self.create_initial_tiles()
        self.subdivide_all()

    def get_statistics(self):
        """Return statistics about the tiling."""
        thin_count = sum(1 for r in self.triangles if r.shape == "thin")
        thick_count = sum(1 for r in self.triangles if r.shape == "thick")
        total = len(self.triangles)

        return {
            'total': total,
            'thin': thin_count,
            'thick': thick_count,
            'thin_ratio': thin_count / total if total > 0 else 0,
            'thick_ratio': thick_count / total if total > 0 else 0,
            'thick_to_thin_ratio': thick_count / thin_count if thin_count > 0 else 0
        }
