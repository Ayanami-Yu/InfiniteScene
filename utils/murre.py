import torch
from utils.scipy import interp_grid


def get_sparse_depth(points, ixt, ext, H, W, use_valid_mask=True):
    """
    Params:
        points: Tensor[N, 3] 3D means of Gaussians.
    Returns:
        Tensor[H, W] The depth values of projected points. Zero if a pixel does not correspond to a 3D point.
    """
    dtype, device = points.dtype, points.device
    ext[:, 1:2] *= -1  # TODO

    coord_cam = points @ ext[:3, :3].T + ext[:3, 3:].T
    coord_img = coord_cam @ ixt.T

    coord_img[:, :2] = coord_img[:, :2] / coord_img[:, 2:]
    x_homo, y_homo = coord_img[:, 0], coord_img[:, 1]
    depths = coord_img[:, 2]

    # TODO valid_idx
    if use_valid_mask:
        x, y = torch.meshgrid(
            torch.arange(W, dtype=dtype, device=device),
            torch.arange(H, dtype=dtype, device=device),
            indexing="xy",
        )
        grid = torch.stack((x, y), dim=-1).reshape(-1, 2)
        valid_mask = torch.nonzero(
            (depths > 0)
            & (x_homo >= 0)
            & (x_homo <= W - 1)
            & (y_homo >= 0)
            & (y_homo <= H - 1),
            as_tuple=False,
        ).squeeze(1)
        coord_img = coord_img[valid_mask, :2]
        sdpt = interp_grid(coord_img, depths[valid_mask].unsqueeze(1), grid, method="nearest").reshape(H, W)

    else:
        x = torch.clamp(coord_img[:, 0], 0, W - 1).int()
        y = torch.clamp(coord_img[:, 1], 0, H - 1).int()

        assert x.shape[0] == y.shape[0] == depths.shape[0]
        sdpt = torch.zeros(H, W, dtype=dtype, device=device)
        sdpt[y, x] = depths

    return sdpt
