
"""
modified open3d azure kinect simple_sample.

Capture and display k4a point clouds using basic open3d functions
"""
import open3d as o3d
import numpy as np
import copy
import traceback
import sys
import ctypes
import os
import pyk4a
from pyk4a import Config, PyK4A
from matplotlib import pyplot as plt
# Add .. to the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import k4a



if __name__ == "__main__":
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            camera_fps=pyk4a.FPS.FPS_5,
            depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()


# From Open3D to numpy
    #np_points = np.asarray(pcd.points)
    capture = k4a.get_capture()
    img_color = capture.color

    azPcd = o3d.geometry.PointCloud()

    azPcd.points = o3d.utility.Vector3dVector(capture.depth_point_cloud.reshape((-1, 3)))
    azPcd.colors =  o3d.utility.Vector3dVector(capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)))


    #source = azPcd.voxel_down_sample(voxel_size=0.002)

    vis = o3d.visualization.Visualizer()

    vis.create_window()
    vis.add_geometry(azPcd)

    while 1:
        capture = k4a.get_capture()
        points = capture.depth_point_cloud.reshape((-1, 3))
        azPcd.points =o3d.utility.Vector3dVector(points)
        azPcd.colors =  o3d.utility.Vector3dVector(capture.transformed_color.reshape((-1, 3)))
        vis.update_geometry(azPcd)
        vis.poll_events()
        vis.update_renderer()
        #o3d.visualization.draw()
        #vis.run()


    plt.imshow(img_color[:, :, 2::-1]) # BGRA to RGB
    plt.show()


                    
