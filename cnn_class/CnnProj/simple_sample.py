
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
    srcr = o3d.io.read_point_cloud("D:/repo/Open3D/examples/test_data/ICP/cloud_bin_0.pcd")

    src = srcr.voxel_down_sample(voxel_size=0.02)
    azPcd = o3d.geometry.PointCloud()

    azPcd.points = o3d.utility.Vector3dVector(capture.depth_point_cloud.reshape((-1, 3)))
    azPcd.colors =  o3d.utility.Vector3dVector(capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)))
    azPcd.estimate_normals()
    points = np.asarray(azPcd.points)
    colors = np.asarray(azPcd.colors)
    azPcd = azPcd.voxel_down_sample(voxel_size=0.02)

    vis = o3d.visualization.Visualizer()

    vis.create_window()
    vis.add_geometry(azPcd)
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    currAzPcd=o3d.geometry.PointCloud()

    threshold = 0.05
    first = True
    while 1:
        capture = k4a.get_capture()

        currAzPcd.points =  o3d.utility.Vector3dVector(capture.depth_point_cloud.reshape((-1, 3)))
        currAzPcd.colors =  o3d.utility.Vector3dVector( capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) /255.0 )
        currAzPcd.estimate_normals()

        #azPcd.points =  o3d.utility.Vector3dVector(capture.depth_point_cloud.reshape((-1, 3)))
        #azPcd.colors =  o3d.utility.Vector3dVector( capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) /255.0 )
        currAzPcd.transform(flip_transform)
        reg_p2l = o3d.pipelines.registration.registration_icp(
            currAzPcd, azPcd, threshold, np.identity(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        currAzPcd.transform(reg_p2l.transformation)
        vis.update_geometry(currAzPcd)


        #currAzPcd = currAzPcd.voxel_down_sample(voxel_size=0.02)
        #reg_p2l = o3d.pipelines.registration.registration_icp( source, target, threshold, np.identity(4),
        #   o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        #   o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        if (first):
            currAzPcd.orient_normals_consistent_tangent_plane(10)
            vis.add_geometry(currAzPcd)
            first=False
        vis.update_geometry(azPcd)
        #vis.update_geometry(currAzPcd)
        vis.poll_events()
        vis.update_renderer()
        azPcd=currAzPcd
        #o3d.visualization.draw()
        #vis.run()

    plt.imshow(img_color[:, :, 2::-1]) # BGRA to RGB
    plt.show()


                    
