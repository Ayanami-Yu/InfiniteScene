import torch
from torch import Tensor


def world_point_to_kth(poses, points, k, device):
    """
    Transforms world coord to the k-th camera coord and then transforms points accordingly
    """
    kth_pose = poses[k]
    inv_kth_pose = torch.inverse(kth_pose)  # [4, 4]

    # transform all camera poses to the k-th camera coord
    new_poses = torch.bmm(inv_kth_pose.unsqueeze(0).expand_as(poses), poses)
    N, W, H, _ = points.shape
    points = points.view(N, W * H, 3)
    homo_points = torch.cat([points, torch.ones(N, W * H, 1).to(device)], dim=-1)
    new_points = inv_kth_pose.unsqueeze(0).expand(N, -1, -1).unsqueeze(
        1
    ) @ homo_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[..., :3].view(N, W, H, _)

    return new_poses, new_points


def world_point_to_obj(poses, points, k, r, elevation, device):
    """
    Transforms world coordinates to be centered around a specific object.

    Args:
        poses (torch.Tensor): Camera to world transformations. Shape: [1, 4, 4].
        points (torch.Tensor): 3D points in world coordinates. Shape: [B, H, W, C].
        k (int): The index of the reference camera.
        r (torch.Tensor): A scalar tensor representing the radius or depth of the object's center
                          from the reference camera.
        elevation (float): The elevation angle (in degrees) to define the new coordinate system's orientation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - new_poses (Tensor[1, 4, 4]): The transformed camera poses in the new object-centric coordinate system.
            - new_points (Tensor[B, H, W, C]): The transformed 3D points in the new object-centric coordinate system.
    """

    # First, transform from the world coordinate system to the coordinate system of the k-th (reference) camera.
    poses, points = world_point_to_kth(poses, points, k, device)

    # Define the pose of the new target coordinate system.
    # The origin will be at the object's center (located at [0, 0, r] in the k-th camera's frame).
    elevation_rad = torch.deg2rad(torch.tensor(180 - elevation)).to(device)
    sin_value_x = torch.sin(elevation_rad)
    cos_value_x = torch.cos(elevation_rad)

    # Rotation matrix for a camera looking at the object from the given elevation.
    R = torch.tensor(
        [
            [1, 0, 0],
            [0, cos_value_x, sin_value_x],
            [0, -sin_value_x, cos_value_x],
        ]
    ).to(device)

    # Translation vector to the object's center.
    t = torch.tensor([0, 0, r]).to(device)

    # Create the 4x4 transformation matrix (pose) for the new object-centric coordinate system.
    pose_obj = torch.eye(4).to(device)
    pose_obj[:3, :3] = R
    pose_obj[:3, 3] = t

    # Transform all points and poses to the target object-centric coord.
    inv_obj_pose = torch.inverse(pose_obj)
    new_poses = torch.bmm(inv_obj_pose.unsqueeze(0).expand_as(poses), poses)
    B, H, W, _ = points.shape
    points = points.view(B, H * W, 3)
    homo_points = torch.cat([points, torch.ones(B, H * W, 1).to(device)], dim=-1)
    new_points = inv_obj_pose.unsqueeze(0).expand(B, -1, -1).unsqueeze(
        1
    ) @ homo_points.unsqueeze(-1)
    new_points = new_points.squeeze(-1)[..., :3].view(B, H, W, _)

    return new_poses, new_points


def sphere2pose(
    c2ws_input: Tensor, theta: Tensor, phi: Tensor, r, device, x=None, y=None
):
    """
    Args:
        c2ws_input (Tensor[B, 4, 4])
    """
    c2ws = c2ws_input.detach().clone()

    # first translate along the z-axis of world coord and then rotate
    c2ws[:, 2, 3] += r
    if x is not None:
        c2ws[:, 1, 3] += y
    if y is not None:
        c2ws[:, 0, 3] += x

    theta = torch.deg2rad(theta).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = (
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, cos_value_x, -sin_value_x, 0],
                [0, sin_value_x, cos_value_x, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    phi = torch.deg2rad(phi).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = (
        torch.tensor(
            [
                [cos_value_y, 0, sin_value_y, 0],
                [0, 1, 0, 0],
                [-sin_value_y, 0, cos_value_y, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )
    c2ws = torch.matmul(rot_mat_x, c2ws)
    c2ws = torch.matmul(rot_mat_y, c2ws)
    return c2ws


def generate_traj_circular(
    c2ws_anchor, H, W, focal, theta, phi, d_r, n_frame, device, dtype=torch.float32
):
    """
    Args:
        c2ws_anchor (Tensor[1, 4, 4])
    """
    # COLMAP coord system: x-right, y-down, z-forward
    assert (
        n_frame % 2 == 1
    ), "n_frame should be an odd number to include the initial (central) camera"
    thetas = torch.linspace(-theta, theta, n_frame, dtype=dtype)
    phis = torch.linspace(-phi, phi, n_frame, dtype=dtype)
    # TODO check whether it's radius or not
    rs = torch.linspace(0, d_r * c2ws_anchor[0, 2, 3], n_frame, dtype=dtype)

    c2ws_list = []
    for th, ph, r in zip(thetas, phis, rs):
        c2w_new = sphere2pose(c2ws_anchor, th, ph, r, device)
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)

    R, T = c2ws[:, :3, :3], c2ws[:, :3, 3:]
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(
        torch.cat(
            (
                new_c2w,
                torch.Tensor([[[0, 0, 0, 1]]])
                .to(device)
                .repeat(new_c2w.shape[0], 1, 1),
            ),
            1,
        )
    )
    # TODO check why conversion is required
    R_new, T_new = (
        w2c[:, :3, :3].permute(0, 2, 1),
        w2c[:, :3, 3],
    )  # convert R to row-major matrix
    image_size = ((H, W),)

    # TODO return camera trajectory
    # return cameras
