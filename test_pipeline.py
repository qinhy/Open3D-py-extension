import time

import numpy as np
import open3dpypro as pro3d


def measure_fps(gen, test_duration=15, func = lambda imgs:None,
                title='#### profile ####'):
    print(title)
    prev_time = time.time()
    start_time = prev_time
    frame_count = 0
    for imgs in gen:
        func(imgs)

        # Compute elapsed time for one frame
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        prev_time = curr_time  # Update for next iteration
        frame_count += 1

        if elapsed_time > 0:
            print(f"Current FPS = {(1.0 / elapsed_time):.2f} FPS         ", end='\r')

        # Check if test duration has passed
        if curr_time - start_time >= test_duration: break

    # Compute average FPS
    print(f"Test completed: Average FPS = {(frame_count/(time.time()-start_time)):.2f}")


def test1():
    
    # def filterNz(pcds_data, pcds_info, meta):
    #     res = []
    #     for i,pcd in enumerate(pcds_data):
    #         pcd = pcd[np.abs(pcd[:,5])>0.5]
    #         res.append(pcd)
    #     return res
    # ld_filterNz = pro3d.processors.Processors.Lambda()
    # ld_filterNz._forward_raw=filterNz


    def filterz(pcds_data, pcds_info, meta):
        res = []
        for i,pcd in enumerate(pcds_data):
            z = pcd[:,2]
            zmean = z.mean()
            pcd = pcd[np.logical_and( z>(zmean-0.5) , z<(zmean+0.5) )]
            res.append(pcd)
        return res
    ld_filterz = pro3d.processors.Processors.Lambda()
    ld_filterz._forward_raw=filterz


    def centerz(pcds_data, pcds_info, meta):
        res = []
        for i,pcd in enumerate(pcds_data):
            pcd[:,2] = pcd[:,2]-pcd[:,2].mean()
            res.append(pcd)
        return res
    ld_centerz = pro3d.processors.Processors.Lambda()
    ld_centerz._forward_raw=centerz

    gen = pro3d.generator.NumpyRawFrameFileGenerator(
                            # sources=['./data/bunny.npy'],
                            sources=['../zed_point_clouds.npy'],
                            shape_types=[pro3d.ShapeType.XYZ])
    
    plane_det = pro3d.processors.Processors.PlaneDetection(distance_threshold=0.05,alpha=0.1)
    n_samples=10_000
    pipes = [
        pro3d.processors.Processors.RandomSample(n_samples=n_samples),
        pro3d.processors.Processors.NumpyToTorch(),
        pro3d.processors.Processors.RadiusSelection(radius=3.0),
        pro3d.processors.Processors.VoxelDownsample(voxel_size=0.05),
        plane_det,
        pro3d.processors.Processors.TorchToNumpy(),
        pro3d.processors.Processors.PlaneNormalize(detection_uuid=plane_det.uuid),
        ld_filterz,

        # vis only
        ld_centerz,
        pro3d.processors.Processors.ZDepthViewer(grid_size=128),
        pro3d.processors.Processors.O3DStreamViewer(),
    ]
    pro3d.processors.PointCloudMatProcessors.validate_once(gen,pipes)
    while len(pipes)>0:
        measure_fps(gen, func=lambda imgs:pro3d.processors.PointCloudMatProcessors.run_once(
                imgs,{},pipes),
                test_duration=100,
                title=f"#### profile ####\n{'  ->  '.join([f.title for f in pipes])}")
        pipes.pop()



if __name__ == "__main__":
    test1()
