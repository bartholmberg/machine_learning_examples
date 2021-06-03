
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
from pyk4a import PyK4A
from matplotlib import pyplot as plt
# Add .. to the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import k4a



if __name__ == "__main__":
    k4a = PyK4A()
    k4a.start()
    pcd = o3d.geometry.PointCloud()
    np_points = np.random.rand(100, 3)

# From numpy to Open3D
    pcd.points = o3d.utility.Vector3dVector(np_points)

# From Open3D to numpy
    np_points = np.asarray(pcd.points)
    capture = k4a.get_capture()
    img_color = capture.color
    #source_raw=capture.depth_point_cloud
    source_raw=capture.depth_point_cloud
    azPcd = o3d.geometry.PointCloud()

    source_raw=source_raw.reshape(source_raw.shape[0]*source_raw.shape[1],3);
    azPcd.points = o3d.utility.Vector3dVector(source_raw)


    #source = azPcd.voxel_down_sample(voxel_size=0.002)

    vis = o3d.visualization.Visualizer()

    vis.create_window()
    vis.add_geometry(azPcd)
    #vis.update_geometry(azPcd)
    #vis.poll_events()
    #vis.update_renderer()
    while 1:
        capture = k4a.get_capture()
        #source_raw=capture.depth_point_cloud
        source_raw=capture.depth_point_cloud
        source_raw=source_raw.reshape(source_raw.shape[0]*source_raw.shape[1],3)
        azPcd.points = o3d.utility.Vector3dVector(source_raw)
        vis.update_geometry(azPcd)
        vis.poll_events()
        vis.update_renderer()
        #o3d.visualization.draw()
        #vis.run()


    plt.imshow(img_color[:, :, 2::-1]) # BGRA to RGB
    plt.show()


                    
