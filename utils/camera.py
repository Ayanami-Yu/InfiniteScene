#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json
import numpy as np
import torch

from scene.cameras import Camera, MiniCam
from utils.general import PILtoTorch
from utils.graphics import (
    fov2focal,
    focal2fov,
    getWorld2View,
    getProjectionMatrix,
    getWorld2View2,
)


WARNED = False


# TODO consider adding z_scale
def load_json(path, H, W):
    cams = []
    with open(path) as json_file:
        contents = json.load(json_file)
        FoVx = contents["camera_angle_x"]
        FoVy = focal2fov(fov2focal(FoVx, W), H)
        zfar = 100.0
        znear = 0.01

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if c2w.shape[0] == 3:
                one = np.zeros((1, 4))
                one[0, -1] = 1
                c2w = np.concatenate((c2w, one), axis=0)

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            w2c = torch.as_tensor(getWorld2View(R, T)).T.cuda()
            proj = getProjectionMatrix(znear, zfar, FoVx, FoVy).T.cuda()
            cams.append(MiniCam(W, H, FoVx, FoVy, znear, zfar, w2c, w2c @ proj))
    return cams


def loadCam(args, id, cam_info, resolution_scale):  # unused
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


# TODO document zfar and znear in JSON file
def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry


def get_render_cam(focalx, c2w, H, W, z_scale=1.0):
    """
    Params:
        c2w: np.ndarray[4, 4] Camera to world transformation.
        z_scale: The scaling factor to multiply zfar and znear.
    """
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    fovx = focal2fov(focalx, W)
    fovy = focal2fov(fov2focal(fovx, W), H)

    znear, zfar = 0.01 * z_scale, 100 * z_scale
    world_view_transform = (
        torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0))
        .transpose(0, 1)
        .cuda()
    )
    projection_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
        .transpose(0, 1)
        .cuda()
    )
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)

    return MiniCam(
        width=W,
        height=H,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )


def get_train_cam(img, focalx, c2w, H, W, white_background: bool, z_scale=1.0, idx=0):
    """
    Params:
        c2w: np.ndarray[4, 4] Camera to world transformation.
        img: The GT image corresponding to that view. That is, the image projected/interpolated from colored point cloud in `generate_pcd`, or the zoomed-in image.
        z_scale: The scaling factor to multiply zfar and znear.
    """
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

    fovx = focal2fov(focalx, W)
    fovy = focal2fov(fov2focal(fovx, W), H)

    img = np.array(img.convert("RGBA")) / 255.0
    img = img[:, :, :3] * img[:, :, 3:4] + bg * (1 - img[:, :, 3:4])
    img = torch.Tensor(img).permute(2, 0, 1)

    return Camera(
        colmap_id=idx,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=img,
        gt_alpha_mask=None,
        image_name="",
        uid=idx,
        data_device="cuda",
        z_scale=z_scale,
    )
