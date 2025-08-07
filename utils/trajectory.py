# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
#
# Copyright 2023 LucidDreamer Authors
#
# Computer Vision Lab, SNU, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from the Computer Vision Lab, SNU or
# its affiliates is strictly prohibited.
#
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
import os
import numpy as np
import torch


def generate_seed(scale, viewangle):
    # World 2 Camera
    #### rotate x,y
    render_poses = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])]
    ang = 5
    for i, j in zip(
        [
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
        ],
        [
            0,
            0,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            0,
            0,
            0,
        ],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.eye(3),
            np.matmul(
                np.array(
                    [
                        [np.cos(th), 0, np.sin(th)],
                        [0, 1, 0],
                        [-np.sin(th), 0, np.cos(th)],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)],
                    ]
                ),
            ),
        )  # Turn left
        posetemp[:3, 3:4] = np.array([0, 0, 0]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    for i, j in zip(
        [
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
        ],
        [0, 0, 0, ang, ang, ang, ang, ang, 0, -ang, -ang, -ang, -ang, -ang, 0, 0, 0, 0],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(-3 * ang / 180 * np.pi), 0, np.sin(-3 * ang / 180 * np.pi)],
                    [0, 1, 0],
                    [
                        -np.sin(-3 * ang / 180 * np.pi),
                        0,
                        np.cos(-3 * ang / 180 * np.pi),
                    ],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [np.cos(th), 0, np.sin(th)],
                        [0, 1, 0],
                        [-np.sin(th), 0, np.cos(th)],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)],
                    ]
                ),
            ),
        )
        posetemp[:3, 3:4] = np.array([1, 0, 0]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    for i, j in zip(
        [
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
        ],
        [0, 0, 0, ang, ang, ang, ang, ang, 0, -ang, -ang, -ang, -ang, -ang, 0, 0, 0, 0],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(3 * ang / 180 * np.pi), 0, np.sin(3 * ang / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(3 * ang / 180 * np.pi), 0, np.cos(3 * ang / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [np.cos(th), 0, np.sin(th)],
                        [0, 1, 0],
                        [-np.sin(th), 0, np.cos(th)],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)],
                    ]
                ),
            ),
        )
        posetemp[:3, 3:4] = np.array([-1, 0, 0]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    # for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]):
    #     th, phi = i/180*np.pi, j/180*np.pi
    #     posetemp = np.zeros((3, 4))
    #     posetemp[:3,:3] = np.matmul(np.eye(3),
    #                                 np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])))
    #     posetemp[:3,3:4] = np.array([0,0,1]).reshape(3,1) # * scale # Transition vector
    #     render_poses.append(posetemp)

    rot_cam = viewangle / 3
    for i, j in zip(
        [
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
        ],
        [
            0,
            0,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            0,
            0,
            0,
        ],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(rot_cam / 180 * np.pi), 0, np.sin(rot_cam / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(rot_cam / 180 * np.pi), 0, np.cos(rot_cam / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [np.cos(th), 0, np.sin(th)],
                        [0, 1, 0],
                        [-np.sin(th), 0, np.cos(th)],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)],
                    ]
                ),
            ),
        )  # Turn left
        posetemp[:3, 3:4] = np.array([0, 0, 0]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    for i, j in zip(
        [
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
        ],
        [0, 0, 0, ang, ang, ang, ang, ang, 0, -ang, -ang, -ang, -ang, -ang, 0, 0, 0, 0],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(rot_cam / 180 * np.pi), 0, np.sin(rot_cam / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(rot_cam / 180 * np.pi), 0, np.cos(rot_cam / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [
                            np.cos(-3 * ang / 180 * np.pi),
                            0,
                            np.sin(-3 * ang / 180 * np.pi),
                        ],
                        [0, 1, 0],
                        [
                            -np.sin(-3 * ang / 180 * np.pi),
                            0,
                            np.cos(-3 * ang / 180 * np.pi),
                        ],
                    ]
                ),
                np.matmul(
                    np.array(
                        [
                            [np.cos(th), 0, np.sin(th)],
                            [0, 1, 0],
                            [-np.sin(th), 0, np.cos(th)],
                        ]
                    ),
                    np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)],
                        ]
                    ),
                ),
            ),
        )
        posetemp[:3, 3:4] = np.array([0, 0, 1]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    for i, j in zip(
        [
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
        ],
        [0, 0, 0, ang, ang, ang, ang, ang, 0, -ang, -ang, -ang, -ang, -ang, 0, 0, 0, 0],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(rot_cam / 180 * np.pi), 0, np.sin(rot_cam / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(rot_cam / 180 * np.pi), 0, np.cos(rot_cam / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [
                            np.cos(3 * ang / 180 * np.pi),
                            0,
                            np.sin(3 * ang / 180 * np.pi),
                        ],
                        [0, 1, 0],
                        [
                            -np.sin(3 * ang / 180 * np.pi),
                            0,
                            np.cos(3 * ang / 180 * np.pi),
                        ],
                    ]
                ),
                np.matmul(
                    np.array(
                        [
                            [np.cos(th), 0, np.sin(th)],
                            [0, 1, 0],
                            [-np.sin(th), 0, np.cos(th)],
                        ]
                    ),
                    np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)],
                        ]
                    ),
                ),
            ),
        )
        posetemp[:3, 3:4] = np.array([0, 0, -1]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    # for i,j in zip([ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,ang,2*ang,3*ang,2*ang,ang,0], [0,0,0,ang,2*ang,3*ang,2*ang,ang,0,-ang,-2*ang,-3*ang,-2*ang,-ang,0,0,0,0]):
    #     th, phi = i/180*np.pi, j/180*np.pi
    #     posetemp = np.zeros((3, 4))
    #     posetemp[:3,:3] = np.matmul(np.array([[np.cos(rot_cam/180*np.pi), 0, np.sin(rot_cam/180*np.pi)], [0, 1, 0], [-np.sin(rot_cam/180*np.pi), 0, np.cos(rot_cam/180*np.pi)]]),
    #                                 np.matmul(np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]), np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])))
    #     posetemp[:3,3:4] = np.array([1,0,0]).reshape(3,1) # * scale # Transition vector
    #     render_poses.append(posetemp)

    rot_cam = viewangle * 2 / 3
    for i, j in zip(
        [
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
        ],
        [
            0,
            0,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            0,
            0,
            0,
        ],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(rot_cam / 180 * np.pi), 0, np.sin(rot_cam / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(rot_cam / 180 * np.pi), 0, np.cos(rot_cam / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [np.cos(th), 0, np.sin(th)],
                        [0, 1, 0],
                        [-np.sin(th), 0, np.cos(th)],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)],
                    ]
                ),
            ),
        )  # Turn left
        posetemp[:3, 3:4] = np.array([0, 0, 0]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    for i, j in zip(
        [
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
        ],
        [0, 0, 0, ang, ang, ang, ang, ang, 0, -ang, -ang, -ang, -ang, -ang, 0, 0, 0, 0],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(rot_cam / 180 * np.pi), 0, np.sin(rot_cam / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(rot_cam / 180 * np.pi), 0, np.cos(rot_cam / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [
                            np.cos(-3 * ang / 180 * np.pi),
                            0,
                            np.sin(-3 * ang / 180 * np.pi),
                        ],
                        [0, 1, 0],
                        [
                            -np.sin(-3 * ang / 180 * np.pi),
                            0,
                            np.cos(-3 * ang / 180 * np.pi),
                        ],
                    ]
                ),
                np.matmul(
                    np.array(
                        [
                            [np.cos(th), 0, np.sin(th)],
                            [0, 1, 0],
                            [-np.sin(th), 0, np.cos(th)],
                        ]
                    ),
                    np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)],
                        ]
                    ),
                ),
            ),
        )
        posetemp[:3, 3:4] = np.array([-1, 0, 0]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    for i, j in zip(
        [
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
        ],
        [0, 0, 0, ang, ang, ang, ang, ang, 0, -ang, -ang, -ang, -ang, -ang, 0, 0, 0, 0],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(rot_cam / 180 * np.pi), 0, np.sin(rot_cam / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(rot_cam / 180 * np.pi), 0, np.cos(rot_cam / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [
                            np.cos(3 * ang / 180 * np.pi),
                            0,
                            np.sin(3 * ang / 180 * np.pi),
                        ],
                        [0, 1, 0],
                        [
                            -np.sin(3 * ang / 180 * np.pi),
                            0,
                            np.cos(3 * ang / 180 * np.pi),
                        ],
                    ]
                ),
                np.matmul(
                    np.array(
                        [
                            [np.cos(th), 0, np.sin(th)],
                            [0, 1, 0],
                            [-np.sin(th), 0, np.cos(th)],
                        ]
                    ),
                    np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)],
                        ]
                    ),
                ),
            ),
        )
        posetemp[:3, 3:4] = np.array([1, 0, 0]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    rot_cam = viewangle
    for i, j in zip(
        [
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
        ],
        [
            0,
            0,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            0,
            0,
            0,
        ],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(rot_cam / 180 * np.pi), 0, np.sin(rot_cam / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(rot_cam / 180 * np.pi), 0, np.cos(rot_cam / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [np.cos(th), 0, np.sin(th)],
                        [0, 1, 0],
                        [-np.sin(th), 0, np.cos(th)],
                    ]
                ),
                np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)],
                    ]
                ),
            ),
        )  # Turn left
        posetemp[:3, 3:4] = np.array([0, 0, 0]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    for i, j in zip(
        [
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
        ],
        [0, 0, 0, ang, ang, ang, ang, ang, 0, -ang, -ang, -ang, -ang, -ang, 0, 0, 0, 0],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(rot_cam / 180 * np.pi), 0, np.sin(rot_cam / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(rot_cam / 180 * np.pi), 0, np.cos(rot_cam / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [
                            np.cos(-3 * ang / 180 * np.pi),
                            0,
                            np.sin(-3 * ang / 180 * np.pi),
                        ],
                        [0, 1, 0],
                        [
                            -np.sin(-3 * ang / 180 * np.pi),
                            0,
                            np.cos(-3 * ang / 180 * np.pi),
                        ],
                    ]
                ),
                np.matmul(
                    np.array(
                        [
                            [np.cos(th), 0, np.sin(th)],
                            [0, 1, 0],
                            [-np.sin(th), 0, np.cos(th)],
                        ]
                    ),
                    np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)],
                        ]
                    ),
                ),
            ),
        )
        posetemp[:3, 3:4] = np.array([0, 0, -1]).reshape(
            3, 1
        )  # * scale # Transition vector
        render_poses.append(posetemp)

    for i, j in zip(
        [
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
            -ang,
            -2 * ang,
            -3 * ang,
            -2 * ang,
            -ang,
            0,
            ang,
            2 * ang,
            3 * ang,
            2 * ang,
            ang,
            0,
        ],
        [0, 0, 0, ang, ang, ang, ang, ang, 0, -ang, -ang, -ang, -ang, -ang, 0, 0, 0, 0],
    ):
        th, phi = i / 180 * np.pi, j / 180 * np.pi
        posetemp = np.zeros((3, 4))
        posetemp[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(rot_cam / 180 * np.pi), 0, np.sin(rot_cam / 180 * np.pi)],
                    [0, 1, 0],
                    [-np.sin(rot_cam / 180 * np.pi), 0, np.cos(rot_cam / 180 * np.pi)],
                ]
            ),
            np.matmul(
                np.array(
                    [
                        [
                            np.cos(3 * ang / 180 * np.pi),
                            0,
                            np.sin(3 * ang / 180 * np.pi),
                        ],
                        [0, 1, 0],
                        [
                            -np.sin(3 * ang / 180 * np.pi),
                            0,
                            np.cos(3 * ang / 180 * np.pi),
                        ],
                    ]
                ),
                np.matmul(
                    np.array(
                        [
                            [np.cos(th), 0, np.sin(th)],
                            [0, 1, 0],
                            [-np.sin(th), 0, np.cos(th)],
                        ]
                    ),
                    np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)],
                        ]
                    ),
                ),
            ),
        )
        posetemp[:3, 3:4] = np.array([0, 0, 1]).reshape(3, 1)  # transition vector
        render_poses.append(posetemp)

    render_poses.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
    render_poses = np.stack(render_poses, axis=0)

    return render_poses


def generate_seed_360(viewangle, n_views, deg_denom=1):
    N = n_views * deg_denom
    render_poses = np.zeros((N, 3, 4))
    for i in range(N):
        th = (viewangle / N) * i / 180 * np.pi
        render_poses[i, :3, :3] = np.array(
            [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
        )
        render_poses[i, :3, 3:4] = np.random.randn(3, 1) * 0.0  # Transition vector

    return render_poses


def generate_seed_360_half(viewangle, n_views):
    N = n_views // 2
    halfangle = viewangle / 2
    render_poses = np.zeros((N * 2, 3, 4))
    for i in range(N):
        th = (halfangle / N) * i / 180 * np.pi
        render_poses[i, :3, :3] = np.array(
            [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
        )
        render_poses[i, :3, 3:4] = np.random.randn(3, 1) * 0.0  # Transition vector
    for i in range(N):
        th = -(halfangle / N) * i / 180 * np.pi
        render_poses[i + N, :3, :3] = np.array(
            [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
        )
        render_poses[i + N, :3, 3:4] = np.random.randn(3, 1) * 0.0  # Transition vector
    return render_poses


def generate_seed_preset(deg_denom=1):
    th_max = 60 / deg_denom
    phi_max = 22.5 / deg_denom
    thlist = np.concatenate(  # in degree
        (
            np.linspace(0, th_max, 4),
            np.linspace(0, -th_max, 4)[1:],
            np.linspace(0, th_max, 4),
            np.linspace(0, -th_max, 4)[1:],
            np.linspace(0, th_max, 4),
            np.linspace(0, -th_max, 4)[1:],
        )
    )
    philist = np.concatenate(
        (
            np.linspace(0, 0, 7),
            np.linspace(-phi_max, -phi_max, 7),
            np.linspace(phi_max, phi_max, 7),
        )
    )
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.zeros((3, 1))

    return render_poses


def generate_seed_newpreset(deg_denom=1):
    th_max = 60 / deg_denom
    phi_max = 22.5 / deg_denom
    thlist = np.concatenate(
        (
            np.linspace(0, th_max, 4),
            np.linspace(0, -th_max, 4)[1:],
            np.linspace(0, th_max, 4),
            np.linspace(0, -th_max, 4)[1:],
        )
    )
    philist = np.concatenate((np.linspace(0, 0, 7), np.linspace(phi_max, phi_max, 7)))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.zeros((3, 1))

    return render_poses


def generate_seed_horizon():
    movement = np.linspace(0, 5, 11)
    render_poses = np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):

        render_poses[i, :3, :3] = np.eye(3)
        render_poses[i, :3, 3:4] = np.array([[-movement[i]], [0], [0]])

    return render_poses


def generate_seed_backward():
    movement = np.linspace(0, 5, 11)
    render_poses = np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        render_poses[i, :3, :3] = np.eye(3)
        render_poses[i, :3, 3:4] = np.array([[0], [0], [movement[i]]])
    return render_poses


def generate_seed_arc():
    degree = 5
    # thlist = np.array([degree, 0, 0, 0, -degree])
    thlist = np.arange(0, degree, 5) + np.arange(0, -degree, 5)[1:]
    phi = 0

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        d = 4.3  # manual central point for arc / you can change this value

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.array(
            [d * np.sin(th / 180 * np.pi), 0, d - d * np.cos(th / 180 * np.pi)]
        ).reshape(3, 1) + np.array(
            [0, d * np.sin(phi / 180 * np.pi), d - d * np.cos(phi / 180 * np.pi)]
        ).reshape(
            3, 1
        )  # Transition vector
        # render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_hemisphere(center_depth, degree=5, deg_denom=1):
    degree = 5 / deg_denom
    thlist = np.array([degree, 0, 0, 0, -degree])
    philist = np.array([0, -degree, 0, degree, 0])
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        d = center_depth  # central point of (hemi)sphere / you can change this value

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.array(
            [d * np.sin(th / 180 * np.pi), 0, d - d * np.cos(th / 180 * np.pi)]
        ).reshape(3, 1) + np.array(
            [0, d * np.sin(phi / 180 * np.pi), d - d * np.cos(phi / 180 * np.pi)]
        ).reshape(
            3, 1
        )  # transition vector

    return render_poses


def generate_seed_hemisphere_(degree, nviews):
    # thlist = np.array([degree, 0, 0, 0, -degree])
    # philist = np.array([0, -degree, 0, degree, 0])
    thlist = degree * np.sin(np.linspace(0, 2 * np.pi, nviews))
    philist = degree * np.cos(np.linspace(0, 2 * np.pi, nviews))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        # curr_pose = np.zeros((1, 3, 4))
        d = 4.3  # manual central point for arc / you can change this value

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.array(
            [d * np.sin(th / 180 * np.pi), 0, d - d * np.cos(th / 180 * np.pi)]
        ).reshape(3, 1) + np.array(
            [0, d * np.sin(phi / 180 * np.pi), d - d * np.cos(phi / 180 * np.pi)]
        ).reshape(
            3, 1
        )  # Transition vector
    return render_poses


def generate_seed_nothing():
    degree = 5
    thlist = np.array([0])
    philist = np.array([0])
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        # curr_pose = np.zeros((1, 3, 4))
        d = 4.3  #

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.array(
            [d * np.sin(th / 180 * np.pi), 0, d - d * np.cos(th / 180 * np.pi)]
        ).reshape(3, 1) + np.array(
            [0, d * np.sin(phi / 180 * np.pi), d - d * np.cos(phi / 180 * np.pi)]
        ).reshape(
            3, 1
        )  # Transition vector
        # render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_lookaround():
    degsum = 60
    thlist = np.concatenate(
        (
            np.linspace(0, degsum, 4),
            np.linspace(0, -degsum, 4)[1:],
            np.linspace(0, degsum, 4),
            np.linspace(0, -degsum, 4)[1:],
            np.linspace(0, degsum, 4),
            np.linspace(0, -degsum, 4)[1:],
        )
    )
    philist = np.concatenate(
        (np.linspace(0, 0, 7), np.linspace(22.5, 22.5, 7), np.linspace(-22.5, -22.5, 7))
    )
    assert len(thlist) == len(philist)

    render_poses = []
    # up / left --> right
    thlist = np.linspace(-degsum, degsum, 2 * degsum + 1)
    for i in range(len(thlist)):
        render_pose = np.zeros((3, 4))
        th = thlist[i]
        phi = 22.5

        render_pose[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_pose[:3, 3:4] = np.zeros((3, 1))
        render_poses.append(render_pose)

    # right / up --> center
    phlist = np.linspace(22.5, 0, 23)
    # Exclude first frame (same as last frame before)
    phlist = phlist[1:]
    for i in range(len(phlist)):
        render_pose = np.zeros((3, 4))
        th = degsum
        phi = phlist[i]

        render_pose[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_pose[:3, 3:4] = np.zeros((3, 1))
        render_poses.append(render_pose)

    # center / right --> left
    thlist = np.linspace(degsum, -degsum, 2 * degsum + 1)
    thlist = thlist[1:]
    for i in range(len(thlist)):
        render_pose = np.zeros((3, 4))
        th = thlist[i]
        phi = 0

        render_pose[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_pose[:3, 3:4] = np.zeros((3, 1))
        render_poses.append(render_pose)

    # left / center --> down
    phlist = np.linspace(0, -22.5, 23)
    phlist = phlist[1:]
    for i in range(len(phlist)):
        render_pose = np.zeros((3, 4))
        th = -degsum
        phi = phlist[i]

        render_pose[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_pose[:3, 3:4] = np.zeros((3, 1))
        render_poses.append(render_pose)

    thlist = np.linspace(-degsum, degsum, 2 * degsum + 1)
    for i in range(len(thlist)):
        render_pose = np.zeros((3, 4))
        th = thlist[i]
        phi = -22.5

        render_pose[:3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_pose[:3, 3:4] = np.zeros((3, 1))
        render_poses.append(render_pose)

    return render_poses


def generate_seed_lookdown():
    degsum = 60
    thlist = np.concatenate(
        (
            np.linspace(0, degsum, 4),
            np.linspace(0, -degsum, 4)[1:],
            np.linspace(0, degsum, 4),
            np.linspace(0, -degsum, 4)[1:],
        )
    )
    philist = np.concatenate((np.linspace(0, 0, 7), np.linspace(-22.5, -22.5, 7)))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.zeros((3, 1))

    return render_poses


def generate_seed_back():
    movement = np.linspace(0, 5, 101)
    render_poses = []  # np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        render_pose = np.zeros((3, 4))
        render_pose[:3, :3] = np.eye(3)
        render_pose[:3, 3:4] = np.array([[0], [0], [movement[i]]])
        render_poses.append(render_pose)

    movement = np.linspace(5, 0, 101)
    movement = movement[1:]
    for i in range(len(movement)):
        render_pose = np.zeros((3, 4))
        render_pose[:3, :3] = np.eye(3)
        render_pose[:3, 3:4] = np.array([[0], [0], [movement[i]]])
        render_poses.append(render_pose)

    return render_poses


def generate_seed_llff(degree, nviews, round=4, d=2.3):
    assert round % 4 == 0
    thlist = degree * np.sin(np.linspace(0, 2 * np.pi * round, nviews))
    philist = degree * np.cos(np.linspace(0, 2 * np.pi * round, nviews))
    zlist = d / 15 * np.sin(np.linspace(0, 2 * np.pi * round // 4, nviews))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        z = zlist[i]

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.array(
            [d * np.sin(th / 180 * np.pi), 0, -z + d - d * np.cos(th / 180 * np.pi)]
        ).reshape(3, 1) + np.array(
            [0, d * np.sin(phi / 180 * np.pi), -z + d - d * np.cos(phi / 180 * np.pi)]
        ).reshape(
            3, 1
        )  # Transition vector
    return render_poses


def generate_seed_headbanging(maxdeg, nviews_per_round, round=3, fullround=1):
    radius = np.concatenate(
        (
            np.linspace(0, maxdeg, nviews_per_round * round),
            maxdeg * np.ones(nviews_per_round * fullround),
            np.linspace(maxdeg, 0, nviews_per_round * round),
        )
    )
    thlist = (
        2.66
        * radius
        * np.sin(
            np.linspace(
                0,
                2 * np.pi * (round + fullround + round),
                nviews_per_round * (round + fullround + round),
            )
        )
    )
    philist = radius * np.cos(
        np.linspace(
            0,
            2 * np.pi * (round + fullround + round),
            nviews_per_round * (round + fullround + round),
        )
    )
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.zeros((3, 1))

    return render_poses


def generate_seed_headbanging_circle(maxdeg, nviews_per_round, round=3, fullround=1):
    radius = np.concatenate(
        (
            np.linspace(0, maxdeg, nviews_per_round * round),
            maxdeg * np.ones(nviews_per_round * fullround),
            np.linspace(maxdeg, 0, nviews_per_round * round),
        )
    )
    thlist = (
        2.66
        * radius
        * np.sin(
            np.linspace(
                0,
                2 * np.pi * (round + fullround + round),
                nviews_per_round * (round + fullround + round),
            )
        )
    )
    philist = radius * np.cos(
        np.linspace(
            0,
            2 * np.pi * (round + fullround + round),
            nviews_per_round * (round + fullround + round),
        )
    )
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]

        render_poses[i, :3, :3] = np.matmul(
            np.array(
                [
                    [np.cos(th / 180 * np.pi), 0, -np.sin(th / 180 * np.pi)],
                    [0, 1, 0],
                    [np.sin(th / 180 * np.pi), 0, np.cos(th / 180 * np.pi)],
                ]
            ),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(phi / 180 * np.pi), -np.sin(phi / 180 * np.pi)],
                    [0, np.sin(phi / 180 * np.pi), np.cos(phi / 180 * np.pi)],
                ]
            ),
        )
        render_poses[i, :3, 3:4] = np.zeros((3, 1))

    return render_poses


def generate_tapered_orbit(  # TODO test
    degree_start=5,
    degree_end=0.5,
    nviews=400,
    rounds=2,
    distance_start=1,
    distance_end=0.1,
    height_decay=0.5,
):
    """
    Generate a camera trajectory where the camera orbits around the target's central axis,
    with orbital and pitch amplitudes gradually decreasing as it moves forward.

    Parameters:
    - degree_start: initial orbital amplitude (degrees)
    - degree_end: final orbital amplitude (degrees)
    - nviews: number of trajectory points
    - rounds: total number of orbital revolutions
    - distance_start: starting camera-to-center distance
    - distance_end: ending camera-to-center distance
    - height_decay: decay coefficient for vertical displacement amplitude over progression

    Returns:
    - render_poses: ndarray of shape (nviews, 3, 4), camera [R|t] poses for each frame
    """
    assert rounds > 0
    # Normalize progress from 0 to 1
    t = np.linspace(0, 1, nviews)

    # Linearly interpolate amplitude and distance
    amp = degree_start * (1 - t) + degree_end * t
    d = distance_start * (1 - t) + distance_end * t

    # Angle sequences: orbit (theta) and pitch (phi)
    theta = amp * np.sin(2 * np.pi * rounds * t)
    phi = amp * np.cos(2 * np.pi * rounds * t)

    # Small vertical oscillation with decay
    z_offset = height_decay * distance_start * np.sin(2 * np.pi * rounds * t) * (1 - t)

    # Initialize pose storage
    render_poses = np.zeros((nviews, 3, 4))

    for i in range(nviews):
        th = np.deg2rad(theta[i])
        ph = np.deg2rad(phi[i])
        Z = z_offset[i]
        Di = d[i]

        # Rotation matrices around Y (yaw) and X (pitch)
        R_y = np.array(
            [
                [np.cos(th), 0, -np.sin(th)],
                [0, 1, 0],
                [np.sin(th), 0, np.cos(th)],
            ]
        )
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(ph), -np.sin(ph)],
                [0, np.sin(ph), np.cos(ph)],
            ]
        )
        R = R_y @ R_x
        render_poses[i, :3, :3] = R

        # Translation components: horizontal, vertical, and z-offset
        t_horiz = np.array([Di * np.sin(th), 0, Di * np.cos(th)])
        t_vert = np.array([0, Di * np.sin(ph), 0])
        t_z = np.array([0, 0, Z])

        render_poses[i, :3, 3] = t_horiz + t_vert + t_z

    return render_poses


def get_pcdGenPoses(pcdgenpath, argdict={}, deg_denom=1):
    if pcdgenpath == "rotate360":
        render_poses = generate_seed_360(360, 10, deg_denom=deg_denom)
    elif pcdgenpath == "lookaround":
        render_poses = generate_seed_preset(deg_denom=deg_denom)
    elif pcdgenpath == "moveright":
        render_poses = generate_seed_horizon()
    elif pcdgenpath == "moveback":
        render_poses = generate_seed_backward()
    elif pcdgenpath == "arc":
        render_poses = generate_seed_arc()
    elif pcdgenpath == "lookdown":
        render_poses = generate_seed_newpreset(deg_denom=deg_denom)
    elif pcdgenpath == "hemisphere":
        render_poses = generate_seed_hemisphere(
            argdict["center_depth"], deg_denom=deg_denom
        )
    else:
        raise ("Invalid pcdgenpath")
    return render_poses


def getCameraPaths():
    """
    Precompute all camera trajectories for rendering videos
    """
    preset_json = {}
    for cam_path in ["back_and_forth", "llff", "headbanging", "tapered"]:  # TODO
        if cam_path == "back_and_forth":
            render_poses = generate_seed_back()
        elif cam_path == "llff":
            render_poses = generate_seed_llff(5, 400, round=4, d=2)
        elif cam_path == "headbanging":
            render_poses = generate_seed_headbanging(
                maxdeg=15, nviews_per_round=180, round=2, fullround=0
            )
        elif cam_path == "tapered":
            render_poses = generate_tapered_orbit()
        else:
            raise ("Unknown pass")

        yz_reverse = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        blender_train_json = {"frames": []}
        for render_pose in render_poses:
            curr_frame = {}
            # Transform world to pixel
            Rw2i = render_pose[:3, :3]
            Tw2i = render_pose[:3, 3:4]

            # Transfrom cam2 to world + change sign of yz axis
            Ri2w = np.matmul(yz_reverse, Rw2i).T
            Ti2w = -np.matmul(Ri2w, np.matmul(yz_reverse, Tw2i))
            Pc2w = np.concatenate((Ri2w, Ti2w), axis=1)
            Pc2w = np.concatenate(
                (Pc2w, np.array([0, 0, 0, 1]).reshape((1, 4))), axis=0
            )

            curr_frame["transform_matrix"] = Pc2w.tolist()
            blender_train_json["frames"].append(curr_frame)

        preset_json[cam_path] = blender_train_json

    return preset_json


def main():
    cam_path = "headbanging_circle"
    os.makedirs("poses_supplementary", exist_ok=True)

    if cam_path == "lookaround":
        render_poses = generate_seed_lookaround()
    elif cam_path == "back":
        render_poses = generate_seed_back()
    elif cam_path == "360":
        render_poses = generate_seed_360(360, 360)
    elif cam_path == "1440":
        render_poses = generate_seed_360(360, 1440)
    elif cam_path == "llff":
        d = 8
        render_poses = generate_seed_llff(5, 400, round=4, d=d)
    elif cam_path == "headbanging":
        round = 3
        render_poses = generate_seed_headbanging(
            maxdeg=15, nviews_per_round=180, round=round, fullround=0
        )
    elif cam_path == "headbanging_circle":
        round = 2
        render_poses = generate_seed_headbanging_circle(
            maxdeg=5, nviews_per_round=180, round=round, fullround=0
        )

    yz_reverse = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    c2w_poses = []
    for render_pose in render_poses:
        ### Transform world to pixel
        Rw2i = render_pose[:3, :3]
        Tw2i = render_pose[:3, 3:4]

        # Transfrom cam2 to world + change sign of yz axis
        Ri2w = np.matmul(yz_reverse, Rw2i).T
        Ti2w = -np.matmul(Ri2w, np.matmul(yz_reverse, Tw2i))
        Pc2w = np.concatenate((Ri2w, Ti2w), axis=1)
        # Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

        c2w_poses.append(Pc2w)

    c2w_poses = np.stack(c2w_poses, axis=0)

    # np.save(f'poses_supplementary/{cam_path}.npy', c2w_poses)

    FX = 5.8269e02
    W = 512
    fov_x = 2 * np.arctan(W / (2 * FX))
    if cam_path in ["360", "1440", "llff", "headbanging"]:
        fov_x = fov_x * 1.2
    blender_train_json = {}
    blender_train_json["camera_angle_x"] = fov_x
    blender_train_json["frames"] = []

    for render_pose in render_poses:
        curr_frame = {}
        ### Transform world to pixel
        Rw2i = render_pose[:3, :3]
        Tw2i = render_pose[:3, 3:4]

        # Transfrom cam2 to world + change sign of yz axis
        Ri2w = np.matmul(yz_reverse, Rw2i).T
        Ti2w = -np.matmul(Ri2w, np.matmul(yz_reverse, Tw2i))
        Pc2w = np.concatenate((Ri2w, Ti2w), axis=1)

        curr_frame["transform_matrix"] = Pc2w.tolist()
        (blender_train_json["frames"]).append(curr_frame)

    import json

    if cam_path == "llff":
        train_json_path = f"poses_supplementary/{cam_path}_d{d}.json"
    elif cam_path == "headbanging":
        train_json_path = f"poses_supplementary/{cam_path}_r{round}.json"
    else:
        train_json_path = f"poses_supplementary/{cam_path}.json"

    with open(train_json_path, "w") as outfile:
        json.dump(blender_train_json, outfile, indent=4)


if __name__ == "__main__":
    main()
