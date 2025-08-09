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
    sdpt = torch.zeros(H, W, dtype=points.dtype, device=points.device)
    for x, y, z in coord_img:
        x, y = int(x), int(y)
        x = min(max(x, 0), W - 1)
        y = min(max(y, 0), H - 1)
        sdpt[y, x] = z
    return sdpt