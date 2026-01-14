"""
Grid-based point sampling utilities for 2D square lattice generation and filtering.

This module provides a Line class for representing 2D lines and functions for
generating regular 2D square lattices (grids) and filtering points based on
geometric constraints between two lines.

Key Features:
- Line class for 2D line representation and geometric operations
- Grid generation with tunable density (int or explicit (nx, ny) tuples)
- Point filtering based on line constraints
- Separate density sampling for inside/outside regions
- Parameter pair generation for scientific applications

Main Functions:
- sample_grid_between_lines(): Standard grid sampling between two lines
- sample_grid_with_separate_densities(): Enhanced sampling with independent 
  density controls for inside and outside regions
- generate_parameter_pairs_grid(): Generate parameter pairs for research
- generate_parameter_pairs_with_separate_densities(): Parameter generation
  with separate density controls
"""

import numpy as np


class Line:
    """
    A class to represent a line in 2D space using the general form: ax + by + c = 0
    """
    def __init__(self, a, b, c):
        """
        Initialize a line with coefficients a, b, c for ax + by + c = 0
        
        Parameters:
        -----------
        a, b, c : float
            Coefficients for the line equation ax + by + c = 0
            Note: a and b cannot both be zero
        """
        self.a = a
        self.b = b
        self.c = c
        
        # Normalize the coefficients for distance calculations
        norm = np.sqrt(a**2 + b**2)
        if norm > 0:
            self.a_norm = a / norm
            self.b_norm = b / norm
            self.c_norm = c / norm
        else:
            raise ValueError("Invalid line: a and b cannot both be zero")
    
    @classmethod
    def from_slope_intercept(cls, slope, intercept):
        """
        Create a line from slope-intercept form: y = mx + b
        
        Parameters:
        -----------
        slope : float
            Slope of the line (m)
        intercept : float
            Y-intercept of the line (b)
        
        Returns:
        --------
        Line : Line object representing y = mx + b
        """
        # Convert y = mx + b to mx - y + b = 0
        return cls(slope, -1, intercept)
    
    @classmethod
    def from_two_points(cls, p1, p2):
        """
        Create a line from two points
        
        Parameters:
        -----------
        p1, p2 : tuple
            Two points (x1, y1) and (x2, y2) that define the line
        
        Returns:
        --------
        Line : Line object passing through both points
        """
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == x2:  # Vertical line
            return cls(1, 0, -x1)
        elif y1 == y2:  # Horizontal line
            return cls(0, 1, -y1)
        else:
            # General case: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x1
            return cls(a, b, c)
    
    def distance_to_point(self, point):
        """
        Calculate the signed distance from a point to the line
        
        Parameters:
        -----------
        point : tuple or array-like
            (x, y) coordinates of the point
        
        Returns:
        --------
        float : Signed distance (positive on one side, negative on the other)
        """
        x, y = point
        return (self.a_norm * x + self.b_norm * y + self.c_norm)
    
    def is_point_above(self, point):
        """
        Check if a point is above the line (positive side)
        
        Parameters:
        -----------
        point : tuple or array-like
            (x, y) coordinates of the point
        
        Returns:
        --------
        bool : True if point is on the positive side of the line
        """
        return self.distance_to_point(point) > 0
    
    def get_y_at_x(self, x):
        """
        Get y-coordinate for a given x-coordinate (if line is not vertical)
        
        Parameters:
        -----------
        x : float
            X-coordinate
        
        Returns:
        --------
        float or None : Y-coordinate, or None if line is vertical
        """
        if abs(self.b) < 1e-10:  # Vertical line
            return None
        return -(self.a * x + self.c) / self.b
    
    def get_x_at_y(self, y):
        """
        Get x-coordinate for a given y-coordinate (if line is not horizontal)
        
        Parameters:
        -----------
        y : float
            Y-coordinate
        
        Returns:
        --------
        float or None : X-coordinate, or None if line is horizontal
        """
        if abs(self.a) < 1e-10:  # Horizontal line
            return None
        return -(self.b * y + self.c) / self.a
    
    def __str__(self):
        return f"Line: {self.a:.2f}x + {self.b:.2f}y + {self.c:.2f} = 0"


def create_grid_points(density, x_range, y_range):
    """
    Create a regular 2D square lattice (grid) of points.
    
    Parameters:
    -----------
    density : int or tuple
        Grid density specification:
        - int: number of points along the shorter dimension
        - tuple (nx, ny): explicit number of points in x and y directions
    x_range : tuple
        (x_min, x_max) for the grid region
    y_range : tuple
        (y_min, y_max) for the grid region
    
    Returns:
    --------
    grid_points : ndarray
        Array of shape (total_points, 2) containing all grid points
    grid_info : dict
        Information about the grid (dimensions, spacing, etc.)
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Determine grid dimensions
    if isinstance(density, tuple):
        nx, ny = density
    else:
        # For rectangular regions, adapt grid to maintain roughly square cells
        x_span = x_max - x_min
        y_span = y_max - y_min
        aspect_ratio = x_span / y_span
        
        if aspect_ratio >= 1:  # Wider than tall
            nx = density
            ny = max(1, int(density / aspect_ratio))
        else:  # Taller than wide
            ny = density
            nx = max(1, int(density * aspect_ratio))
    
    # Create 1D arrays for each dimension
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    
    # Create 2D meshgrid and flatten to get all points
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Calculate grid information
    grid_spacing_x = (x_max - x_min) / (nx - 1) if nx > 1 else 0
    grid_spacing_y = (y_max - y_min) / (ny - 1) if ny > 1 else 0
    
    grid_info = {
        'grid_dimensions': (nx, ny),
        'total_grid_points': len(grid_points),
        'grid_spacing': (grid_spacing_x, grid_spacing_y),
        'x_coords': x_coords,
        'y_coords': y_coords
    }
    
    return grid_points, grid_info


def point_between_lines(point, line1, line2):
    """
    Check if a point lies between two lines.
    A point is between two lines if it's on opposite sides of each line.
    
    Parameters:
    -----------
    point : tuple or array-like
        (x, y) coordinates of the point
    line1, line2 : Line objects
        The two boundary lines
    
    Returns:
    --------
    bool
        True if the point is between the lines, False otherwise
    """
    d1 = line1.distance_to_point(point)
    d2 = line2.distance_to_point(point)
    return d1 * d2 <= 0  # Different signs or one is zero


def filter_points_between_lines(grid_points, line1, line2):
    """
    Filter grid points to keep only those lying between two lines.
    
    Parameters:
    -----------
    grid_points : ndarray
        Array of shape (n_points, 2) containing grid points
    line1, line2 : Line objects
        The two lines that define the region boundaries
    
    Returns:
    --------
    valid_points : ndarray
        Array containing only points between the lines
    point_mask : ndarray
        Boolean mask indicating which original points are valid
    """
    if len(grid_points) == 0:
        return np.array([]).reshape(0, 2), np.array([], dtype=bool)
    
    # Check which points lie between the lines
    point_mask = np.array([point_between_lines(point, line1, line2) for point in grid_points])
    valid_points = grid_points[point_mask]
    
    return valid_points, point_mask


def sample_grid_between_lines(line1, line2, density=20, x_range=(-5, 5), y_range=(-5, 5)):
    """
    Create a 2D square lattice and filter points to lie between two lines.
    
    This is the main interface function for grid-based point sampling.
    It generates a regular 2D square lattice and filters points based on
    geometric constraints defined by two boundary lines.
    
    Parameters:
    -----------
    line1, line2 : Line objects
        The two lines that define the region boundaries
    density : int or tuple
        Grid density specification:
        - int: number of points along the shorter dimension  
        - tuple (nx, ny): explicit number of points in x and y directions
    x_range : tuple
        (x_min, x_max) for the grid region
    y_range : tuple
        (y_min, y_max) for the grid region
    
    Returns:
    --------
    valid_points : ndarray
        Array of shape (n_valid, 2) containing points between the lines
    grid_info : dict
        Comprehensive information about the grid and filtering results:
        - 'grid_dimensions': (nx, ny) tuple
        - 'total_grid_points': total number of lattice points
        - 'valid_points': number of points between the lines
        - 'invalid_points': number of points outside the region
        - 'coverage_ratio': fraction of valid points
        - 'grid_spacing': (dx, dy) spacing between grid points
        - 'all_grid_points': all lattice points before filtering
        - 'point_mask': boolean mask for valid points
    """
    # Step 1: Create the complete 2D square lattice
    all_grid_points, basic_grid_info = create_grid_points(density, x_range, y_range)
    
    # Step 2: Filter points to keep only those between the lines
    valid_points, point_mask = filter_points_between_lines(all_grid_points, line1, line2)
    
    # Step 3: Compile comprehensive grid information
    grid_info = {
        **basic_grid_info,
        'all_grid_points': all_grid_points,
        'valid_points': len(valid_points),
        'invalid_points': len(all_grid_points) - len(valid_points),
        'coverage_ratio': len(valid_points) / len(all_grid_points) if len(all_grid_points) > 0 else 0,
        'point_mask': point_mask
    }
    
    return valid_points, grid_info


def sample_grid_with_separate_densities(line1, line2, 
                                       inside_density=(20, 20), outside_density=(10, 10), 
                                       x_range=(-5, 5), y_range=(-5, 5),
                                       sample_both=True):
    """
    Create grids with separate density controls for inside and outside the line region.
    
    This function provides independent density control for sampling points between
    two boundary lines (inside region) and outside those lines (outside region).
    Both density parameters must be specified as explicit (nx, ny) tuples.
    
    Parameters:
    -----------
    line1, line2 : Line objects
        The two lines that define the region boundaries
    inside_density : tuple
        Grid density for points between the lines:
        - tuple (nx, ny): explicit number of points in x and y directions
    outside_density : tuple
        Grid density for points outside the lines:
        - tuple (nx, ny): explicit number of points in x and y directions
    x_range : tuple
        (x_min, x_max) for the grid region
    y_range : tuple
        (y_min, y_max) for the grid region
    sample_both : bool
        If True, sample both inside and outside regions
        If False, sample only the inside region (equivalent to sample_grid_between_lines)
    
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'inside_points': ndarray of points between the lines
        - 'outside_points': ndarray of points outside the lines (if sample_both=True)
        - 'all_points': ndarray of all sampled points
        - 'grid_info': comprehensive information about both grids
    
    Example:
    --------
    >>> line1 = Line.from_slope_intercept(slope=1, intercept=0)
    >>> line2 = Line.from_slope_intercept(slope=-1, intercept=2)
    >>> result = sample_grid_with_separate_densities(
    ...     line1, line2, 
    ...     inside_density=(30, 30), outside_density=(10, 10),
    ...     x_range=(-3, 3), y_range=(-3, 3),
    ...     sample_both=True
    ... )
    >>> print(f"Inside: {len(result['inside_points'])}, Outside: {len(result['outside_points'])}")
    """
    result = {
        'inside_points': np.array([]).reshape(0, 2),
        'outside_points': np.array([]).reshape(0, 2),
        'all_points': np.array([]).reshape(0, 2),
        'grid_info': {}
    }
    
    # Step 1: Create grid for inside region (between lines)
    inside_grid_points, inside_grid_info = create_grid_points(inside_density, x_range, y_range)
    inside_points, inside_mask = filter_points_between_lines(inside_grid_points, line1, line2)
    
    # For inside sampling, we only keep the points that are actually between the lines
    result['inside_points'] = inside_points
    
    if sample_both:
        # Step 2: Create grid for outside region
        outside_grid_points, outside_grid_info = create_grid_points(outside_density, x_range, y_range)
        _, outside_between_mask = filter_points_between_lines(outside_grid_points, line1, line2)
        
        # For outside sampling, we keep points that are NOT between the lines
        outside_mask = ~outside_between_mask
        outside_points = outside_grid_points[outside_mask]
        result['outside_points'] = outside_points
        
        # Combine all points
        if len(inside_points) > 0 and len(outside_points) > 0:
            result['all_points'] = np.vstack([inside_points, outside_points])
        elif len(inside_points) > 0:
            result['all_points'] = inside_points
        elif len(outside_points) > 0:
            result['all_points'] = outside_points
        
        # Create comprehensive grid info
        result['grid_info'] = {
            'inside_grid_info': {
                **inside_grid_info,
                'sampled_points': len(inside_points),
                'total_grid_points': len(inside_grid_points),
                'efficiency': len(inside_points) / len(inside_grid_points) if len(inside_grid_points) > 0 else 0
            },
            'outside_grid_info': {
                **outside_grid_info,
                'sampled_points': len(outside_points),
                'total_grid_points': len(outside_grid_points),
                'efficiency': len(outside_points) / len(outside_grid_points) if len(outside_grid_points) > 0 else 0
            },
            'summary': {
                'total_inside_points': len(inside_points),
                'total_outside_points': len(outside_points),
                'total_sampled_points': len(result['all_points']),
                'inside_density_spec': inside_density,
                'outside_density_spec': outside_density,
                'region_bounds': {'x_range': x_range, 'y_range': y_range}
            }
        }
    else:
        # Only inside sampling
        result['all_points'] = inside_points
        result['grid_info'] = {
            'inside_grid_info': {
                **inside_grid_info,
                'sampled_points': len(inside_points),
                'total_grid_points': len(inside_grid_points),
                'efficiency': len(inside_points) / len(inside_grid_points) if len(inside_grid_points) > 0 else 0
            },
            'summary': {
                'total_inside_points': len(inside_points),
                'total_outside_points': 0,
                'total_sampled_points': len(inside_points),
                'inside_density_spec': inside_density,
                'outside_density_spec': 'N/A (not sampled)',
                'region_bounds': {'x_range': x_range, 'y_range': y_range}
            }
        }
    
    return result


def subsample_points(points, subsample_factor):
    """
    Subsample points by a given factor while maintaining grid structure where possible.
    
    Parameters:
    -----------
    points : ndarray
        Array of shape (n_points, 2) containing points
    subsample_factor : float
        Factor by which to reduce the number of points (0 < factor <= 1)
        
    Returns:
    --------
    subsampled_points : ndarray
        Subsampled points
    """
    if subsample_factor >= 1.0:
        return points
    
    n_points = len(points)
    n_keep = max(1, int(n_points * subsample_factor))
    
    # For systematic subsampling, take every nth point
    step = max(1, n_points // n_keep)
    indices = np.arange(0, n_points, step)[:n_keep]
    
    return points[indices]


def generate_parameter_pairs_grid(beta_range, anneal_range, density=20):
    """
    Generate parameter pairs using grid-based sampling between boundary lines.
    
    This function creates a 2D grid of (BETA_1, ANNEAL_PARAM) pairs by defining
    the parameter space with boundary lines and sampling points within that region.
    
    Parameters:
    -----------
    beta_range : tuple
        (beta_min, beta_max) range for BETA_1 parameter
    anneal_range : tuple
        (anneal_min, anneal_max) range for ANNEAL_PARAM parameter
    density : int, optional
        Grid density for sampling (default: 20)
    
    Returns:
    --------
    parameter_pairs : list of tuples
        List of (BETA_1, ANNEAL_PARAM) pairs
    grid_info : dict
        Information about the grid sampling
    
    Example:
    --------
    >>> pairs, info = generate_parameter_pairs_grid(
    ...     beta_range=(0.5, 3.0), 
    ...     anneal_range=(0.1, 0.9), 
    ...     density=15
    ... )
    >>> print(f"Generated {len(pairs)} parameter pairs")
    """
    beta_min, beta_max = beta_range
    anneal_min, anneal_max = anneal_range
    
    # Define boundary lines for rectangular parameter space
    # For a rectangular region, we use lines that form the boundaries
    line1 = Line.from_slope_intercept(slope=0, intercept=anneal_min)    # bottom boundary
    line2 = Line.from_slope_intercept(slope=0, intercept=anneal_max)    # top boundary
    
    # Sample grid points within the parameter space
    points, grid_info = sample_grid_between_lines(
        line1, line2, 
        density=density,
        x_range=beta_range,
        y_range=anneal_range
    )
    
    # Convert points to parameter pairs
    parameter_pairs = [(float(point[0]), float(point[1])) for point in points]
    
    return parameter_pairs, grid_info


def generate_parameter_pairs_with_separate_densities(beta_range, anneal_range, 
                                                   inside_density=(20, 20), outside_density=(10, 10),
                                                   boundary_lines=None, sample_both=True):
    """
    Generate parameter pairs using separate density controls for different regions.
    
    This function extends the basic grid sampling to allow independent density
    controls for different regions of the parameter space, defined by boundary lines.
    
    Parameters:
    -----------
    beta_range : tuple
        (beta_min, beta_max) range for BETA_1 parameter
    anneal_range : tuple
        (anneal_min, anneal_max) range for ANNEAL_PARAM parameter
    inside_density : tuple
        Grid density for points between the boundary lines:
        - tuple (nx, ny): explicit number of points in x and y directions
    outside_density : tuple
        Grid density for points outside the boundary lines:
        - tuple (nx, ny): explicit number of points in x and y directions
    boundary_lines : tuple of Line objects or None
        (line1, line2) defining the region boundaries. If None, creates
        horizontal lines at 1/3 and 2/3 of the anneal range
    sample_both : bool
        If True, sample both inside and outside regions
        If False, sample only the inside region
    
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'inside_pairs': list of (beta, anneal) pairs in the inside region
        - 'outside_pairs': list of (beta, anneal) pairs in the outside region
        - 'all_pairs': list of all parameter pairs
        - 'grid_info': comprehensive sampling information
    
    Example:
    --------
    >>> # High density in a specific region, low density elsewhere
    >>> result = generate_parameter_pairs_with_separate_densities(
    ...     beta_range=(0.5, 3.0),
    ...     anneal_range=(0.1, 0.9),
    ...     inside_density=(30, 20),  # High density in region of interest
    ...     outside_density=(10, 10), # Low density elsewhere
    ...     sample_both=True
    ... )
    >>> print(f"Inside: {len(result['inside_pairs'])}, Outside: {len(result['outside_pairs'])}")
    """
    # Define default boundary lines if none provided
    if boundary_lines is None:
        anneal_min, anneal_max = anneal_range
        anneal_span = anneal_max - anneal_min
        # Create horizontal lines at 1/3 and 2/3 of anneal range
        line1 = Line.from_slope_intercept(slope=0, intercept=anneal_min + anneal_span/3)
        line2 = Line.from_slope_intercept(slope=0, intercept=anneal_min + 2*anneal_span/3)
    else:
        line1, line2 = boundary_lines
    
    # Use the separate density sampling function
    sampling_result = sample_grid_with_separate_densities(
        line1, line2,
        inside_density=inside_density,
        outside_density=outside_density,
        x_range=beta_range,
        y_range=anneal_range,
        sample_both=sample_both
    )
    
    # Convert points to parameter pairs
    inside_pairs = [(float(point[0]), float(point[1])) for point in sampling_result['inside_points']]
    outside_pairs = [(float(point[0]), float(point[1])) for point in sampling_result['outside_points']]
    all_pairs = [(float(point[0]), float(point[1])) for point in sampling_result['all_points']]
    
    result = {
        'inside_pairs': inside_pairs,
        'outside_pairs': outside_pairs,
        'all_pairs': all_pairs,
        'grid_info': sampling_result['grid_info'],
        'boundary_lines': (line1, line2)
    }
    
    return result


def generate_triangular_parameter_pairs(vertex1, vertex2, vertex3, density=20):
    """
    Generate parameter pairs within a triangular region using grid sampling.
    
    Parameters:
    -----------
    vertex1, vertex2, vertex3 : tuple
        Three vertices (beta, anneal_param) defining the triangle
    density : int, optional
        Grid density for sampling (default: 20)
    
    Returns:
    --------
    parameter_pairs : list of tuples
        List of (BETA_1, ANNEAL_PARAM) pairs within the triangle
    grid_info : dict
        Information about the grid sampling
    
    Example:
    --------
    >>> # Define a triangular parameter space
    >>> pairs, info = generate_triangular_parameter_pairs(
    ...     vertex1=(0.5, 0.1),  # (beta, anneal)
    ...     vertex2=(3.0, 0.5),  
    ...     vertex3=(1.5, 0.9),
    ...     density=15
    ... )
    """
    # Find bounding box for the triangle
    betas = [v[0] for v in [vertex1, vertex2, vertex3]]
    anneals = [v[1] for v in [vertex1, vertex2, vertex3]]
    
    beta_range = (min(betas), max(betas))
    anneal_range = (min(anneals), max(anneals))
    
    # Create lines for each edge of the triangle
    line1 = Line.from_two_points(vertex1, vertex2)
    line2 = Line.from_two_points(vertex2, vertex3)
    line3 = Line.from_two_points(vertex3, vertex1)
    
    # Generate grid points in the bounding box
    all_points, basic_info = create_grid_points(density, beta_range, anneal_range)
    
    # Filter points to be inside all three triangle edges
    valid_points = []
    for point in all_points:
        # Check if point is on the correct side of all three triangle edges
        # This is a simplified approach - for proper triangle containment,
        # you might want to use barycentric coordinates or similar
        d1 = line1.distance_to_point(point)
        d2 = line2.distance_to_point(point)
        d3 = line3.distance_to_point(point)
        
        # Check if point is inside triangle (this is a basic implementation)
        # You may need to adjust the signs based on triangle orientation
        if d1 >= 0 and d2 >= 0 and d3 >= 0:
            valid_points.append(point)
        elif d1 <= 0 and d2 <= 0 and d3 <= 0:
            valid_points.append(point)
    
    valid_points = np.array(valid_points)
    
    grid_info = {
        **basic_info,
        'valid_points': len(valid_points),
        'triangle_vertices': [vertex1, vertex2, vertex3],
        'coverage_ratio': len(valid_points) / len(all_points) if len(all_points) > 0 else 0
    }
    
    parameter_pairs = [(float(point[0]), float(point[1])) for point in valid_points]
    
    return parameter_pairs, grid_info


if __name__ == "__main__":
    # Example usage and testing
    print("Testing grid sampling utilities...")
    
    # Test Line class
    line1 = Line.from_slope_intercept(slope=1, intercept=0)   # y = x
    line2 = Line.from_slope_intercept(slope=-1, intercept=2)  # y = -x + 2
    
    print(f"Line 1: {line1}")
    print(f"Line 2: {line2}")
    
    # Test standard grid sampling
    print("\n" + "="*60)
    print("Testing standard grid sampling...")
    points, info = sample_grid_between_lines(
        line1, line2, density=10, x_range=(-2, 2), y_range=(-2, 2)
    )
    
    print(f"Grid dimensions: {info['grid_dimensions']}")
    print(f"Total grid points: {info['total_grid_points']}")
    print(f"Valid points: {info['valid_points']}")
    print(f"Coverage ratio: {info['coverage_ratio']:.3f}")
    
    # Test separate density sampling
    print("\n" + "="*60)
    print("Testing separate density sampling...")
    
    # Example 1: High density inside, low density outside
    result1 = sample_grid_with_separate_densities(
        line1, line2,
        inside_density=(20, 20), outside_density=(8, 8),
        x_range=(-2, 2), y_range=(-2, 2),
        sample_both=True
    )
    
    print(f"\n1. High inside density (20,20), low outside density (8,8):")
    print(f"   Inside points: {len(result1['inside_points'])}")
    print(f"   Outside points: {len(result1['outside_points'])}")
    print(f"   Total points: {len(result1['all_points'])}")
    
    inside_info = result1['grid_info']['inside_grid_info']
    outside_info = result1['grid_info']['outside_grid_info']
    print(f"   Inside efficiency: {inside_info['efficiency']:.3f}")
    print(f"   Outside efficiency: {outside_info['efficiency']:.3f}")
    
    # Example 2: Different aspect ratios
    result2 = sample_grid_with_separate_densities(
        line1, line2,
        inside_density=(15, 25), outside_density=(30, 10),
        x_range=(-2, 2), y_range=(-2, 2),
        sample_both=True
    )
    
    print(f"\n2. Different aspect ratios - Inside: (15,25), Outside: (30,10):")
    print(f"   Inside points: {len(result2['inside_points'])}")
    print(f"   Outside points: {len(result2['outside_points'])}")
    print(f"   Total points: {len(result2['all_points'])}")
    
    # Test parameter pair generation
    print("\n" + "="*60)
    print("Testing parameter pair generation...")
    
    param_pairs, param_info = generate_parameter_pairs_grid(
        beta_range=(0.5, 3.0),
        anneal_range=(0.1, 0.9),
        density=8
    )
    
    print(f"Generated {len(param_pairs)} parameter pairs")
    print(f"Sample pairs: {param_pairs[:3]}")
    
    # Test separate density parameter pair generation
    print("\n" + "="*60)
    print("Testing separate density parameter pair generation...")
    
    param_result = generate_parameter_pairs_with_separate_densities(
        beta_range=(0.5, 3.0),
        anneal_range=(0.1, 0.9),
        inside_density=(15, 12), outside_density=(8, 6),
        sample_both=True
    )
    
    print(f"Inside parameter pairs: {len(param_result['inside_pairs'])}")
    print(f"Outside parameter pairs: {len(param_result['outside_pairs'])}")
    print(f"Total parameter pairs: {len(param_result['all_pairs'])}")
    print(f"Sample inside pairs: {param_result['inside_pairs'][:3]}")
    print(f"Sample outside pairs: {param_result['outside_pairs'][:3]}")
    
    print("\n" + "="*60)
    print("Grid sampling utilities test completed!")
    print("="*60)