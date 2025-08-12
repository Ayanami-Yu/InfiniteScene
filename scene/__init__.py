###
# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
# All rights reserved.
###
import random

from arguments import GSParams
from scene.dataset_readers import readDataInfo
from scene.gaussian_model import GaussianModel


class Scene:
    gaussians: GaussianModel

    def __init__(
        self, traindata, gaussians: GaussianModel, opt: GSParams, init_pcd_path=None
    ):
        self.traindata = traindata
        self.gaussians = gaussians

        info = readDataInfo(traindata, opt.white_background, init_pcd_path)
        random.shuffle(info.train_cameras)  # multi-res consistent random shuffling
        self.cameras_extent = info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = info.train_cameras
        print("Loading Preset Cameras")
        self.preset_cameras = {}
        for campath in info.preset_cameras.keys():
            self.preset_cameras[campath] = info.preset_cameras[
                campath
            ]  # a list of MiniCams

        self.gaussians.create_from_pcd(info.point_cloud, self.cameras_extent)
        self.gaussians.training_setup(opt)

    def getTrainCameras(self):
        return self.train_cameras

    def getPresetCameras(self, preset):
        """
        Returns: A list of MiniCams corresponding to the trajectory specified by `preset`.
        """
        assert preset in self.preset_cameras
        return self.preset_cameras[preset]
