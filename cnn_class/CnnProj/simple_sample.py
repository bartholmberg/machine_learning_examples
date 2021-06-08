
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
            depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED ,
            synchronized_images_only=True,
        )
    )
    k4a.start()
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    capture = k4a.get_capture()
    img_color = capture.color
    srcr = o3d.io.read_point_cloud("D:/repo/Open3D/examples/test_data/ICP/cloud_bin_0.pcd")

    src = srcr.voxel_down_sample(voxel_size=0.02)
    azPcd = o3d.geometry.PointCloud()

    azPcd.points = o3d.utility.Vector3dVector(capture.depth_point_cloud.reshape((-1, 3)))
    azPcd.colors =  o3d.utility.Vector3dVector(capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)))
    
    azPcd.estimate_normals()

    #azPcd.orient_normals_consistent_tangent_plane(100)
    #azPcd.estimate_normals()
    #points = np.asarray(azPcd.points)
    #colors = np.asarray(azPcd.colors)
    #azPcd = azPcd.voxel_down_sample(voxel_size=0.02)
    azPcd.transform(flip_transform)
    #o3d.visualization.draw_geometries([azPcd], point_show_normal=True)
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    #vis.add_geometry(azPcd)

    nxtAzPcd=o3d.geometry.PointCloud()
    vis.add_geometry(nxtAzPcd)
    threshold = 0.05
    first = True
    while 1:
        capture = k4a.get_capture()

        nxtAzPcd.points =  o3d.utility.Vector3dVector(capture.depth_point_cloud.reshape((-1, 3)))
        nxtAzPcd.colors =  o3d.utility.Vector3dVector( capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) /255.0 )
        nxtAzPcd.estimate_normals()
        #nxtAzPcd = nxtAzPcd.voxel_down_sample(voxel_size=0.02)
        #azPcd.points =  o3d.utility.Vector3dVector(capture.depth_point_cloud.reshape((-1, 3)))
        #azPcd.colors =  o3d.utility.Vector3dVector( capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) /255.0 )
        nxtAzPcd.transform(flip_transform)
        reg_p2l = o3d.pipelines.registration.registration_icp(
            nxtAzPcd, azPcd,threshold, np.identity(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        nxtAzPcd.transform(reg_p2l.transformation)
        vis.update_geometry(azPcd)


        #nxtAzPcd = nxtAzPcd.voxel_down_sample(voxel_size=0.02)
        #reg_p2l = o3d.pipelines.registration.registration_icp( source, target, threshold, np.identity(4),
        #   o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        #   o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        if (first):
            #nxtAzPcd.orient_normals_consistent_tangent_plane(10)
            vis.add_geometry(nxtAzPcd)
            first=False
        #vis.update_geometry(azPcd)
        #vis.update_geometry(currAzPcd)
        vis.poll_events()
        vis.update_renderer()
        azPcd=nxtAzPcd
        #o3d.visualization.draw()
        #vis.run()

    plt.imshow(img_color[:, :, 2::-1]) # BGRA to RGB
    plt.show()


                    
