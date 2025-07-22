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


def test1(sources):

    ld_centerz = pro3d.processors.Processors.Lambda(uuid='Lambda:ld_centerz',save_results_to_meta=True)
    def centerz(pcds_data, pcds_info, meta):
        res = []
        for i,pcd in enumerate(pcds_data):
            z_mean = -pcd[:,2].mean()
            pcd[:,2] = pcd[:,2] + z_mean
            res.append(pcd)
            # Construct a 4x4 transformation matrix T for the Z-shift
            T = np.eye(4)
            T[2, 3] = z_mean
            ld_centerz.forward_T.append(T.tolist())
        return res
    ld_centerz._forward_raw=centerz

    ld_filterz = pro3d.processors.Processors.Lambda()
    def filterz(pcds_data, pcds_info, meta):
        res = []
        for i,pcd in enumerate(pcds_data):
            z = pcd[:,2]
            zmedian = ld_filterz._mat_funcs[i].median(z)
            pcd = pcd[ld_filterz._mat_funcs[i].logical_and( z>(zmedian-0.5) , z<(zmedian+0.5) )]

            z = pcd[:,2]
            zmean = ld_filterz._mat_funcs[i].mean(z)
            # pcd = pcd[ld_filterz._mat_funcs[i].logical_or( z<(zmean-0.05) , z>(zmean+0.05) )]
            pcd = pcd[ z<(zmean-0.05)]

            res.append(pcd)
        return res
    ld_filterz._forward_raw=filterz
    
    ld_filterNz = pro3d.processors.Processors.Lambda()
    def filterNz(pcds_data, pcds_info, meta):
        res = []
        for i,pcd in enumerate(pcds_data):
            # z = pcd[:,2]
            # zmean = ld_filterz._mat_funcs[i].mean(z)
            z = pcd[:,2]
            zmedian = ld_filterz._mat_funcs[i].median(z)
            c = ld_filterz._mat_funcs[i].logical_and(ld_filterz._mat_funcs[i].abs(pcd[:,5])<0.90,z<zmedian)
            c = ld_filterz._mat_funcs[i].logical_or(c, z>zmedian)
            pcd = pcd[c]
            res.append(pcd)
        return res
    ld_filterNz._forward_raw=filterNz

    # ld_filterxy = pro3d.processors.Processors.Lambda()
    # def filterxy(pcds_data, pcds_info, meta):
    #     res = []
    #     for i,pcd in enumerate(pcds_data):
    #         x = pcd[:,0]
    #         y = pcd[:,1]
    #         zmedian = ld_filterxy._mat_funcs[i].median(z)
    #         pcd = pcd[ld_filterxy._mat_funcs[i].logical_and( z>(zmedian-0.5) , z<(zmedian+0.5) )]
    #         res.append(pcd)
    #     return res
    # ld_filterxy._forward_raw=filterxy

    ld_backward_Ts = pro3d.processors.Processors.Lambda()
    def backward_Ts(pcds_data, pcds_info, meta):
        res = []
        forwardTs = []
        for i in ["PlaneNormalize:pn",]:
            pass

        for i,pcd in enumerate(pcds_data):
            pass
        return res    
    ld_backward_Ts._forward_raw=backward_Ts


    n_samples=40_000
    voxel_size=0.02

    gen = pro3d.generator.NumpyRawFrameFileGenerator(sources=sources,
                            shape_types=[pro3d.ShapeType.XYZ])    
    plane_det = pro3d.processors.Processors.PlaneDetection(distance_threshold=voxel_size*2,alpha=0.1)
    pipes = [
        pro3d.processors.Processors.RandomSample(n_samples=n_samples),
        pro3d.processors.Processors.NumpyToTorch(),
        
        #### GPU
        pro3d.processors.Processors.RadiusSelection(radius=3.0),
        pro3d.processors.Processors.VoxelDownsample(voxel_size=voxel_size),
        plane_det,
        pro3d.processors.Processors.PlaneNormalize(uuid="PlaneNormalize:pn",
                                detection_uuid=plane_det.uuid,save_results_to_meta=True),        
        # pro3d.processors.Processors.TorchNormals(k=20),

        ld_filterz,
        ld_centerz,
        # ld_filterNz,
        #### end GPU

        pro3d.processors.Processors.TorchToNumpy(),
        pro3d.processors.Processors.O3DStreamViewer(),

        # pro3d.processors.Processors.CPUNormals(),
        pro3d.processors.Processors.SimpleSegConnectedComponents(minpoints=50,thickness=1.0,top_n=5,resolution=voxel_size),
        pro3d.processors.Processors.ZDepthViewer(grid_size=32,img_size=128),
    ]
    pro3d.processors.PointCloudMatProcessors.validate_once(gen,pipes)
    while len(pipes)>0:
        measure_fps(gen, func=lambda imgs:pro3d.processors.PointCloudMatProcessors.run_once(
                imgs,{},pipes),
                test_duration=20,
                title=f"#### profile ####\n{'  ->  '.join([f.title for f in pipes])}")
        return
        pipes.pop()



if __name__ == "__main__":
    for s in [
               ['../zed_point_clouds.npy'],
               ['./data/bunny.npy'],
            ]:
        print(s)
        test1(s)
