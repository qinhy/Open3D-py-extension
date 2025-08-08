import math
import time
from typing import List, Union

import cv2
import numpy as np
import torch
from image_pipeblocks.image_pipeblocks.ImageMat import ColorType, ImageMat, ImageMatInfo, ShapeType
from image_pipeblocks.image_pipeblocks.processors import Processors as ImgProcessors
import open3dpypro as pro3d
from open3dpypro.PointCloudMat import MatOps, PointCloudMat, PointCloudMatInfo


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
    grid_size:int=-1
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
    
    # --------------------------------------------------------------------- #
    def imgBackToPCD(
        self,
        img: Union[np.ndarray, torch.Tensor],
        funcs: MatOps,
    ) -> np.ndarray:
        # Get the indices where the matrix is not zero
        y_indices, x_indices = np.nonzero(img) # (mat != 0).nonzero(as_tuple=True)
        # Get the corresponding pixel values
        values = img[y_indices, x_indices]
        # Combine x, y, and value
        pcd_r = funcs.stack((x_indices, y_indices, values),dim=1)
        return pcd_r
    
    def pcd2img(
        self,
        pcd: Union[np.ndarray, torch.Tensor],
        funcs: MatOps,
    ) -> Union[np.ndarray, torch.Tensor]:
        
        is_torch  = torch.is_tensor(pcd)   
        device    = pcd.device if is_torch else "cpu"

        if pcd is None or len(pcd) == 0:
            # empty => zero image
            gs = 128 if self.grid_size < 0 else self.grid_size
            if is_torch:
                depth = funcs.zeros((1,1, gs, gs), dtype=funcs.float16)
            else:
                depth = funcs.zeros((gs, gs), dtype=funcs.uint8)
            return depth

        grid_size = int(len(pcd)** 0.5) if self.grid_size < 0 else self.grid_size
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

        if not (x_max - x_min < 1e-5 or y_max - y_min < 1e-5 or z_max - z_min < 1e-5):
            x_norm = (x - x_min) / (x_max - x_min)
            y_norm = (y - y_min) / (y_max - y_min)

            # Map to grid indices
            xi = funcs.clip(
                       funcs.astype_int32(x_norm * (grid_size - 1)), 0, grid_size - 1)
            yi = funcs.clip(
                       funcs.astype_int32(y_norm * (grid_size - 1)), 0, grid_size - 1)

            z_img = funcs.zeros((grid_size, grid_size), dtype=funcs.float16, device=device)

            # Assign z to grid cells — here, use last point if collisions
            z = funcs.astype_float16(z)
            z = (z - z_min) / (z_max - z_min)
            z_img[yi, xi] = z
            z_img[z_img == 0] = z_img[z_img > 0].mean()

            if not is_torch:
                z_img = funcs.astype_uint8(z_img * 255)
                if self.img_size:
                    z_img = cv2.resize(z_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            else:
                z_img = z_img.unsqueeze(0).unsqueeze(0)
                if self.img_size:
                    z_img = torch.nn.functional.interpolate(z_img, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return z_img

    def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo] = [], meta={}) -> List[np.ndarray]:
        self._img_data = []
        for i, pcd in enumerate(pcds_data):
            z_img = self.pcd2img(pcd, self._mat_funcs[i])
            self._img_data.append(z_img)
        return self._img_data


def test1(sources,loop=False):

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

        meta[ld_direction.uuid] = directions
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

    ld_filter50cm = pro3d.processors.Processors.Lambda()
    def filter50cm(pcds_data, pcds_info, meta):
        res = []
        for i,pcd in enumerate(pcds_data):
            z = pcd[:,2]
            zmedian = ld_filter50cm._mat_funcs[i].median(z)
            pcd = pcd[ld_filter50cm._mat_funcs[i].logical_and( z>(zmedian-0.5) , z<(zmedian+0.5) )]
            res.append(pcd)
        return res
    ld_filter50cm._forward_raw=filter50cm
    
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
            c = ~ld_filterNz._mat_funcs[i].logical_and(ld_filterNz._mat_funcs[i].abs(pcd[:,5])<0.90,z<zmedian)
            # c = ld_filterNz._mat_funcs[i].logical_or(c, z>zmedian)
            pcd = pcd[c]
            res.append(pcd)
        return res
    ld_filterNz._forward_raw=filterNz

    mergedepth = pro3d.processors.Processors.Lambda()
    def merged(pcds_data, pcds_info, meta):
        fu = meta["ZDepthViewer:full"]._img_data
        seg = meta["ZDepthViewer:seg"]._img_data
        for i in seg:
            im = np.hstack([fu[0],i])
            cv2.imwrite(f'./tmp/{int(time.time()*1000)}.png',im)
        return pcds_data
    mergedepth._forward_raw=merged

    binimg_cls = ImgProcessors.Lambda(title='seg',out_color_type=ColorType.BGR)
    def cleanandfit(imgs_data: List[np.ndarray], imgs_info: List[ImageMatInfo]=[], meta={}):
        res = []
        min_area = 1000
        directions = meta.get(ld_direction.uuid, [])

        for idx,binary in enumerate(imgs_data):
            # Ensure single-channel uint8
            if binary.dtype != np.uint8:
                binary = binary.astype(np.uint8)

            # Threshold (binary 0/255)
            _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

            ###############################################################            
            # Step 1: Fill holes inside regions
            # Invert mask to find background-connected components
            holes = cv2.bitwise_not(binary)
            # Flood fill from top-left corner
            h, w = holes.shape
            flood = np.zeros((h+2, w+2), np.uint8)  # Padding needed for floodFill
            cv2.floodFill(holes, flood, (0, 0), 255)
            # Invert flood-filled to get only the holes
            holes_filled = cv2.bitwise_not(holes)
            # Combine with original mask
            filled_mask = cv2.bitwise_or(binary, holes_filled)
            # Step 2: Separate connected regions (optional erosion/dilation)
            kernel = np.ones((3,3), np.uint8)
            # Slight erosion to break thin connections
            separated = cv2.erode(filled_mask, kernel, iterations=2)
            # Optional: re-dilate to recover size (if needed)
            binary = cv2.dilate(separated, kernel, iterations=2)
            # Step 3: Label each region (if needed)
            # num_labels, labels = cv2.connectedComponents(separated)
            ###############################################################

            # Connected components + stats (fast) — no need to relabel later
            # labels: int32 matrix, stats: [label, x, y, w, h, area]
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels <= 1:
                # No components found, return original binary
                output_image = np.zeros_like(binary, dtype=np.uint8)
                res.append(output_image)
                continue

            # Keep only components >= min_area (exclude background label 0)
            keep = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area)[0]
            keep = keep[keep != 0]

            # Build filtered binary via label-index lookup (no Python loops)
            keep_mask = np.zeros(num_labels, dtype=bool)
            keep_mask[keep] = True

            ###############################################################       
            # Prepare 3-channel output (copy of filtered mask)
            filtered_mask = keep_mask[labels]     # boolean mask of kept components
            filtered_binary = (filtered_mask * 255).astype(np.uint8)
            output_image = np.dstack([filtered_binary, filtered_binary, filtered_binary])
            # output_image = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)

            # For each remaining component, compute centers per column/row in vectorized form
            for lbl in keep:
                x = stats[lbl, cv2.CC_STAT_LEFT]
                y = stats[lbl, cv2.CC_STAT_TOP]
                w = stats[lbl, cv2.CC_STAT_WIDTH]
                h = stats[lbl, cv2.CC_STAT_HEIGHT]

                region_mask = (labels[y:y+h, x:x+w] == lbl)
                if not region_mask.any(): continue

                # Precompute index grids for this region
                row_idx = np.arange(h, dtype=np.int64)[:, None]   # shape (h,1)
                col_idx = np.arange(w, dtype=np.int64)[None, :]   # shape (1,w)

                # ---- X-direction: center per column (mark GREEN) ----
                col_counts = region_mask.sum(axis=0)                               # (w,)
                valid_cols = col_counts > 0
                if valid_cols.any() and  "_x" in directions[idx]:
                    # sum of row indices per column; integer division matches int(np.mean(...))
                    sum_rows_per_col = (row_idx * region_mask).sum(axis=0)         # (w,)
                    cy = (sum_rows_per_col[valid_cols] // col_counts[valid_cols])  # (k,)
                    cx = np.where(valid_cols)[0]                                   # (k,)
                    output_image[y + cy, x + cx] = (255, 0, lbl)

                # ---- Y-direction: center per row (mark RED) ----
                row_counts = region_mask.sum(axis=1)                               # (h,)
                valid_rows = row_counts > 0
                if valid_rows.any() and  "_y" in directions[idx]:
                    sum_cols_per_row = (col_idx * region_mask).sum(axis=1)         # (h,)
                    cx2 = (sum_cols_per_row[valid_rows] // row_counts[valid_rows]) # (m,)
                    cy2 = np.where(valid_rows)[0]                                  # (m,)
                    output_image[y + cy2, x + cx2] = (0, 255, lbl)

            res.append(output_image)
            # res+= [output_image[..., 0], output_image[..., 1]]            
            ###############################################################
            # directions = meta.get(ld_direction.uuid, [])
            # if len(directions) > 0:
            #     # Draw direction arrows on the output image
            #     for j, direction in enumerate(directions):
            #         if "_x" in direction:
            #             tmp = output_image[..., 0]
            #         elif "_y" in direction:
            #             tmp = output_image[..., 1]
            #     res.append(tmp)
        return res
    binimg_cls._forward_raw=cleanandfit


    ld_back2Pcd = pro3d.processors.Processors.Lambda()
    def back2Pcd(imgs_data, imgs_info, meta):
        res = []
        forwardTs = []
        for i in ["PlaneNormalize:pn",]:
            pass

        for i,pcd in enumerate(imgs_data):
            pass
        return res    
    ld_back2Pcd._forward_raw=back2Pcd


    n_samples=50_000
    voxel_size=0.01
    radius=2.0

    gen = pro3d.generator.NumpyRawFrameFileGenerator(sources=sources,loop=loop,
                            shape_types=[pro3d.ShapeType.XYZ])    
    plane_det = pro3d.processors.Processors.PlaneDetection(distance_threshold=voxel_size*2,alpha=0.1)

    pip_gpu_pcd_filters = [
        pro3d.processors.Processors.RandomSample(n_samples=n_samples),

        pro3d.processors.Processors.NumpyToTorch(),        
        #### GPU
        pro3d.processors.Processors.RadiusSelection(radius=radius),
        pro3d.processors.Processors.VoxelDownsample(voxel_size=voxel_size),
        plane_det,
        pro3d.processors.Processors.PlaneNormalize(uuid="PlaneNormalize:pn",
                                detection_uuid=plane_det.uuid,save_results_to_meta=True),
        ld_centerz,
        ld_filter50cm,            
        ld_direction,
    ]

    pip_gpu_zdepth = [
        ZDepthImage(grid_size=-1),
        ImgProcessors.TorchResize(target_size=(224, 224)),
    ]

    pip_gpu_seg = [
        ImgProcessors.SegmentationModelsPytorch(ckpt_path='./tmp/epoch=183-step=58144.ckpt',
                #'./tmp/epoch=2-step=948.ckpt',#'./tmp/epoch=183-step=58144.ckpt',epoch=92-step=36735.ckpt
                device='cuda:0',encoder_name='timm-efficientnet-b8',encoder_weights='imagenet'),
        ImgProcessors.TorchGrayToNumpyGray(),
        ImgProcessors.CvImageViewer(title='segdepth',scale=2.0),
        binimg_cls,
    ]

    pipe_vis_depth = [
        ImgProcessors.TorchGrayToNumpyGray(),
        ImgProcessors.CvImageViewer(title='depth',scale=2.0),
    ]
    
    pipe_vis_segdepth = [
        ImgProcessors.CvImageViewer(title='finallines',scale=2.0),
    ]

    pipe_post_pcd = [
        pro3d.processors.Processors.TorchToNumpy(),
        pro3d.processors.Processors.O3DStreamViewer(),
    ]
    
    pipes=[pip_gpu_pcd_filters, pip_gpu_zdepth, pipe_vis_depth, pipe_vis_segdepth, pip_gpu_seg, pipe_post_pcd]
    def run_once(imgs,meta={},
            pipes=pipes,
            validate=False):
        pip_gpu_pcd_filters, pip_gpu_zdepth, pipe_vis_depth,  pipe_vis_segdepth, pip_gpu_seg, pipe_post_pcd = pipes
        try:
            for fn in pip_gpu_pcd_filters:
                imgs,meta = (fn.validate if validate else fn)(imgs,meta)
            
            pcds = [i.copy() for i in imgs]

            for fn in pip_gpu_zdepth:
                imgs,meta = (fn.validate if validate else fn)(imgs,meta)            
            
            segimgs = [i.copy() for i in imgs]
            for fn in pip_gpu_seg:
                segimgs,meta = (fn.validate if validate else fn)(segimgs,meta)

            for fn in pipe_vis_depth:
                imgs,meta = (fn.validate if validate else fn)(imgs,meta)
                
            for fn in pipe_vis_segdepth:
                segimgs,meta = (fn.validate if validate else fn)(segimgs,meta)
                
            for fn in pipe_post_pcd:
                pcds,meta = (fn.validate if validate else fn)(pcds,meta)
        except Exception as e:
            print(f"Error in processing {fn.uuid}: {e}")
            raise e
        return imgs,meta
    
    for imgs in gen:
        run_once(imgs,validate=True)
        break

    while len(pipes)>0:
        measure_fps(gen, func=lambda imgs:run_once(imgs,{},pipes),
                test_duration=20,
                title=f"#### profile ####\n{'  ->  '.join([f.title for ff in pipes for f in ff])}")
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
        test1(s,loop=True)
