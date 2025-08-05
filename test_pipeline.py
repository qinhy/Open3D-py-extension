import math
import time
from typing import List

import cv2
import numpy as np
from image_pipeblocks.image_pipeblocks.ImageMat import ColorType, ImageMat, ShapeType
from image_pipeblocks.image_pipeblocks.processors import Processors as ImgProcessors
import open3dpypro as pro3d
from open3dpypro.PointCloudMat import PointCloudMat, PointCloudMatInfo


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


class ZDepthImage(pro3d.processors.Processors.Lambda):    
    title: str = 'z_depth_img'
    bg:tuple[int,int,int] = (0,0,0) # (125,125,125)
    grid_size: int = 256  # Grid resolution (e.g., 256 x 256)
    img_size:int=224
    _img_data:list=[]
    inf:float=math.inf
    x_min:float=math.inf
    x_max:float=math.inf
    y_min:float=math.inf
    y_max:float=math.inf
    z_min:float=math.inf
    z_max:float=math.inf

    def model_post_init(self, context):
        return super().model_post_init(context)

    def validate_pcd(self, idx, pcd: PointCloudMat):
        if pcd.is_ndarray():
            pcd.require_ndarray()
        if pcd.is_torch_tensor():
            pcd.require_torch_float()
        self.init_common_utility_methods(idx,pcd.is_ndarray())

    def build_out_mats(self, validated_pcds, converted_raw_pcds):
        self.out_mats = [
            ImageMat(color_type=ColorType.GRAYSCALE).build(pcd)
            for old, pcd in zip(validated_pcds, converted_raw_pcds)
        ]
        return self.out_mats

    def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo] = [], meta={}) -> List[np.ndarray]:
        self._img_data = []
        for i, pcd in enumerate(pcds_data):
            z_img_color = np.zeros((self.grid_size,self.grid_size,3),dtype=np.uint8)
            if len(pcd)>0:
                # Use XYZ coordinates only
                xyz = pcd[:, :3]
                # Extract Z (depth) channel
                x = self._mat_funcs[0].copy_mat(xyz[:, 0])
                y = self._mat_funcs[0].copy_mat(xyz[:, 1])
                z = self._mat_funcs[0].copy_mat(xyz[:, 2])

                # Normalize x and y to [0, 1]
                x_min = x.min() if self.x_min==self.inf else self.x_min
                x_max = x.max() if self.x_max==self.inf else self.x_max
                y_min = y.min() if self.y_min==self.inf else self.y_min
                y_max = y.max() if self.y_max==self.inf else self.y_max
                z_min = z.min() if self.z_min==self.inf else self.z_min
                z_max = z.max() if self.z_max==self.inf else self.z_max
                # xy_min, xy_max = min(x_min,y_min), max(x_max,y_max)
                print(x_min,y_min,x_max,y_max,z_min,z_max)

                if not (x_max - x_min < 1e-5 or y_max - y_min < 1e-5):
                    # logger(f"Skipping point cloud {i}: flat x or y range.")
                    # continue
                    # x_norm = (x - xy_min) / (xy_max - xy_min)
                    # y_norm = (y - xy_min) / (xy_max - xy_min)
                    x_norm = (x - x_min) / (x_max - x_min)
                    y_norm = (y - y_min) / (y_max - y_min)

                    # Map to grid indices
                    xi = np.clip((x_norm * (self.grid_size - 1)).astype(np.int32), 0, self.grid_size - 1)
                    yi = np.clip((y_norm * (self.grid_size - 1)).astype(np.int32), 0, self.grid_size - 1)

                    # Initialize grid with NaNs (or large value)
                    z_grid = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)

                    # Assign z to grid cells — here, use last point if collisions
                    z_grid[yi, xi] = z

                    # Fill NaNs with min value (optional, or use interpolation)
                    if np.isnan(z_grid).any():
                        z_grid = np.where(np.isnan(z_grid), z_min, z_grid)

                    if z_max - z_min < 1e-5:
                        z_norm = np.zeros_like(z_grid)
                    else:
                        z_norm = (z_grid - z_min) / (z_max - z_min)

                    z_img = (z_norm * 255).astype(np.uint8)
                    # Optional: apply colormap
                    z_img_color = cv2.applyColorMap(z_img, cv2.COLORMAP_JET)
                    z_img_color[z_img==0] = self.bg
                    z_img_color[:,:,1] = z_img

                    if self.img_size:
                        z_img_color = cv2.resize(z_img_color,(self.img_size,self.img_size))
                        z_img = cv2.resize(z_img,(self.img_size,self.img_size))
                    self._img_data.append(z_img)
        return self._img_data

    

def test1(sources):

    ld_direction = pro3d.processors.Processors.Lambda()
    def direction(pcds_data, pcds_info, meta):
        directions = []

        for pcd in pcds_data:
            x, y = pcd[:, 0], pcd[:, 1]
            counts = {
                "minus_x": (x < 0).sum(),
                "plus_x": (x > 0).sum(),
                "minus_y": (y < 0).sum(),
                "plus_y": (y > 0).sum(),
            }
            dominant = max(counts, key=counts.get)
            directions.append(dominant)
            print(dominant)

        meta["directions"] = directions
        return pcds_data

    ld_direction._forward_raw=direction
    
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

    ld_filterz_1 = pro3d.processors.Processors.Lambda()
    def filterz(pcds_data, pcds_info, meta):
        res = []
        for i,pcd in enumerate(pcds_data):
            z = pcd[:,2]
            zmedian = ld_filterz_1._mat_funcs[i].median(z)
            pcd = pcd[ld_filterz_1._mat_funcs[i].logical_and( z>(zmedian-0.5) , z<(zmedian+0.5) )]
            res.append(pcd)
        return res
    ld_filterz_1._forward_raw=filterz
    
    ld_filterz_2 = pro3d.processors.Processors.Lambda()
    def filterz(pcds_data, pcds_info, meta):
        res = []
        for i,pcd in enumerate(pcds_data):
            z = pcd[:,2]
            zmean = ld_filterz_2._mat_funcs[i].mean(z)
            pcd = pcd[ z<(zmean-0.025)]

            res.append(pcd)
        return res
    ld_filterz_2._forward_raw=filterz
    
    ld_filterNz = pro3d.processors.Processors.Lambda()
    def filterNz(pcds_data, pcds_info, meta):
        res = []
        for i,pcd in enumerate(pcds_data):
            # z = pcd[:,2]
            # zmean = ld_filterz._mat_funcs[i].mean(z)
            z = pcd[:,2]
            zmedian = ld_filterNz._mat_funcs[i].median(z)
            c = ld_filterNz._mat_funcs[i].logical_and(ld_filterNz._mat_funcs[i].abs(pcd[:,5])<0.90,z<zmedian)
            c = ld_filterNz._mat_funcs[i].logical_or(c, z>zmedian)
            pcd = pcd[c]
            res.append(pcd)
        return res
    ld_filterNz._forward_raw=filterNz

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


    mergedepth = pro3d.processors.Processors.Lambda()
    def merged(pcds_data, pcds_info, meta):
        fu = meta["ZDepthViewer:full"]._img_data
        seg = meta["ZDepthViewer:seg"]._img_data
        for i in seg:
            im = np.hstack([fu[0],i])
            cv2.imwrite(f'./tmp/{int(time.time()*1000)}.png',im)
        return pcds_data
    mergedepth._forward_raw=merged

    n_samples=50_000
    voxel_size=0.02
    radius=2.0
    top_n=5

    gen = pro3d.generator.NumpyRawFrameFileGenerator(sources=sources,loop=False,
                            shape_types=[pro3d.ShapeType.XYZ])    
    plane_det = pro3d.processors.Processors.PlaneDetection(distance_threshold=voxel_size*2,alpha=0.1)
    pipes = [
        pro3d.processors.Processors.RandomSample(n_samples=n_samples),

        pro3d.processors.Processors.NumpyToTorch(),        
        #### GPU
        pro3d.processors.Processors.RadiusSelection(radius=radius),
        pro3d.processors.Processors.VoxelDownsample(voxel_size=voxel_size),
        plane_det,
        pro3d.processors.Processors.PlaneNormalize(uuid="PlaneNormalize:pn",
                                detection_uuid=plane_det.uuid,save_results_to_meta=True),
        # pro3d.processors.Processors.TorchNormals(k=20),
        ld_centerz,
        ld_filterz_1,
        # ld_filterNz,
        #### end GPU
            
        ld_direction,
        pro3d.processors.Processors.TorchToNumpy(),
        pro3d.processors.Processors.O3DStreamViewer(),
        
        # pro3d.processors.Processors.ZDepthViewer(uuid="ZDepthViewer:full",grid_size=64,img_size=512,
        #                             x_min=-radius,x_max=radius,y_min=-radius,y_max=radius,z_min=-0.50,z_max=0.50,
        #                             save_results_to_meta=True),
        
        # ld_filterz_2,

        # pro3d.processors.Processors.CPUNormals(),
        # pro3d.processors.Processors.SimpleSegConnectedComponents(
        #                 minpoints=50,thickness=1.0,top_n=top_n,resolution=voxel_size),

        # pro3d.processors.Processors.ZDepthViewer(uuid="ZDepthViewer:seg",grid_size=64,img_size=512,
        #                             x_min=-radius,x_max=radius,y_min=-radius,y_max=radius,z_min=-0.50,z_max=0.50,
        #                             save_results_to_meta=True),
        ZDepthImage(grid_size=128,img_size=224),
                    # x_min=-radius,x_max=radius,y_min=-radius,y_max=radius,z_min=-0.50,z_max=0.50,),
        ImgProcessors.CvImageViewer(),
        # pro3d.processors.Processors.MergePCDs(),
        # pro3d.processors.Processors.O3DStreamViewer(),
        # mergedepth,s
    ]
    pro3d.processors.PointCloudMatProcessors.validate_once(gen,pipes)
    while len(pipes)>0:
        measure_fps(gen, func=lambda imgs:pro3d.processors.PointCloudMatProcessors.run_once(
                imgs,{},pipes),
                test_duration=200,
                title=f"#### profile ####\n{'  ->  '.join([f.title for f in pipes])}")
        return
        pipes.pop()



if __name__ == "__main__":
    for s in [
               ['../zed_point_clouds.npy'],
            #    ['./data/bunny.npy'],
               ['../2025-06-12_12-10-25.white.top.right.Csv.npy'],
               ['../2025-06-12_12-10-25.white.top.center.Csv.npy'],
               ['../2025-06-12_11-58-20.black.top.right.Csv.npy'],
               ['../2025-06-12_11-58-20.black.top.center.Csv.npy'],
            ]:
        print(s)
        test1(s)
