import torch

# from scipy.interpolate import griddata
# from scipy.ndimage import minimum_filter as min_filter, maximum_filter as max_filter
import cupy as cp
from cupyx.scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from cupyx.scipy.ndimage import (
    minimum_filter as min_filter,
    maximum_filter as max_filter,
)


def interp_grid(points, values, grid, method="linear", fill_value=0):
    """
    Simple wrapper for griddata using torch tensors
    Params:
        points: tensor of shape [N, 2] (in camera coord)
        values: tensor of shape [N, 3] (the colors of each point)
        grid: the points that the interpolator evaluates at
    """
    # points_np = points.detach().cpu().numpy()
    points_cp = cp.asarray(points.detach())
    # values_np = values.detach().cpu().numpy()
    values_cp = cp.asarray(values.detach())
    # grid_np = grid.detach().cpu().numpy()
    grid_cp = cp.asarray(grid.detach())

    if method == "linear":
        interpolator = LinearNDInterpolator(points_cp, values_cp, fill_value=fill_value)
    elif method == "nearest":
        interpolator = NearestNDInterpolator(points_cp, values_cp)
    else:
        raise ValueError("Unrecognized interpolation method")

    return torch.as_tensor(interpolator(grid_cp), dtype=values.dtype)


def minimum_filter(input, size, axes=(0, 1)):
    """Simple wrapper for minimum_filter using torch tensors"""
    # input_np = input.detach().cpu().numpy()
    input_cp = cp.asarray(input.detach())
    return torch.as_tensor(min_filter(input_cp, size=size, axes=axes))


def maximum_filter(input, size, axes=(0, 1)):
    """Simple wrapper for maximum_filter using torch tensors"""
    # input_np = input.detach().cpu().numpy()
    input_cp = cp.asarray(input.detach())
    return torch.as_tensor(max_filter(input_cp, size=size, axes=axes))
