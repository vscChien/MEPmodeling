import numpy as np
from scipy.interpolate import pchip_interpolate

def gen_resistance_mono(p):
    """
    Generate membrane resistances of 100 motoneurons (R1 > R2 >...> R100).
    
    Parameters:
    p (array_like): Input parameters (at least 5 elements).
    
    Returns:
    numpy.ndarray: Resistances for 100 motoneurons.
    """
    # Ensure p is a flat array (equivalent to p(:) in MATLAB)
    p = np.asarray(p).flatten()
    
    # Define indices for MN[1, 10, 20, 60, 100]
    n = np.array([1, 10, 20, 60, 100])
    
    # Calculate R of MN[1, 10, 20, 60, 100]
    # Note: MATLAB p(1:5) corresponds to Python p[:5]
    R_points = np.flipud(np.cumsum(p[:5]))
    
    # Interpolate for motoneurons 1 to 100 using Piecewise Cubic Hermite Interpolating Polynomial
    query_points = np.arange(1, 101)
    R = pchip_interpolate(n, R_points, query_points)
    
    return R