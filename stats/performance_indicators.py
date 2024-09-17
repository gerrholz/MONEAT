from copy import deepcopy
from typing import Callable, List

import numpy as np
import numpy.typing as npt

from pymoo.indicators.hv import HV

def hypervolume(refpoint: np.ndarray, points: List[npt.ArrayLike]):
    """
    Calculate the hypervolume of a set of points with respect to a reference point.
    
    Parameters:
    -----------
    refpoint: np.ndarray
        The reference point for the hypervolume calculation.
    points: List[npt.ArrayLike]
        A list of points to calculate the hypervolume for.
    """
    return HV(ref_point=refpoint*-1)(np.array(points)*-1)

def sparsity(front: List[npt.ArrayLike]):
    """
    Calculate the sparsity of a set of points.
    
    Parameters:
    -----------
    front: List[npt.ArrayLike]
        A list of points to calculate the sparsity for.
    """
    if len(front) < 2:
        return 0.0
    
    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objectives = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(front)):
            sparsity_value += objectives[i] - objectives[i-1]

    sparsity_value /= len(front) -1 
    return sparsity_value


def cardinality(front: List[npt.ArrayLike]):
    """
    Calculate the cardinality of a set of points.
    
    Parameters:
    -----------
    front: List[npt.ArrayLike]
        A list of points to calculate the cardinality for.
    """
    # Filter out duplicates
    front = np.array(front)
    front = np.unique(front, axis=0)
    return len(front)


