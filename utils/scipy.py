import torch
from scipy.interpolate import griddata
from scipy.ndimage import minimum_filter as min_filter, maximum_filter as max_filter

def interp_grid(points, values, grid, method='linear', fill_value=0):
    """Simple wrapper for griddata using torch tensors"""
    points_np = points.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()
    grid_np = grid.detach().cpu().numpy()

    return torch.tensor(griddata(points_np, values_np, grid_np, method=method, fill_value=fill_value), dtype=values.dtype, device=values.device)


def minimum_filter(input, size, axes=(0, 1)):
    """Simple wrapper for minimum_filter using torch tensors"""
    input_np = input.detach().cpu().numpy()
    return torch.tensor(min_filter(input_np, size, axes), dtype=input.dtype, device=input.device)


def maximum_filter(input, size, axes=(0, 1)):
    """Simple wrapper for maximum_filter using torch tensors"""
    input_np = input.detach().cpu().numpy()
    return torch.tensor(max_filter(input_np, size, axes), dtype=input.dtype, device=input.device)