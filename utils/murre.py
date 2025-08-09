import torch


def get_sparse_depth(points, ixt, ext, H, W):
    """
    Params:
        points: Tensor[N, 3] 3D means of Gaussians.
    Returns:
        Tensor[H, W] The depth values of projected points. Zero if a pixel does not correspond to a 3D point.
    """
    ext[:, 1:2] *= -1  # TODO

    coord_cam = points @ ext[:3, :3].T + ext[:3, 3:].T
    coord_img = coord_cam @ ixt.T

    coord_img[:, :2] = coord_img[:, :2] / coord_img[:, 2:]
    x_coords = torch.clamp(coord_img[:, 0], 0, W - 1).long()
    y_coords = torch.clamp(coord_img[:, 1], 0, H - 1).long()
    depths = coord_img[:, 2]

    sdpt = torch.zeros(H, W, dtype=points.dtype, device=points.device)
    sdpt[y_coords, x_coords] = depths
    return sdpt