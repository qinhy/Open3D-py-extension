import math
import time
from typing import List, Union

import cv2
import numpy as np
import torch
from image_pipeblocks.image_pipeblocks.ImageMat import ColorType, ImageMat, ImageMatInfo
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
    z_mean:float=math.inf

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
    
    # # --------------------------------------------------------------------- #
    # def imgBackToPCD(
    #     self,
    #     idx: int,
    #     img: Union[np.ndarray, torch.Tensor],
    #     funcs: MatOps,
    # ) -> np.ndarray:
    #     # Get the indices where the matrix is not zero
    #     y_indices, x_indices = funcs.nonzero(img) # (mat != 0).nonzero(as_tuple=True)
    #     # Get the corresponding pixel values
    #     values = img[y_indices, x_indices]
    #     # Combine x, y, and value
    #     pcd_r = funcs.stack((x_indices, y_indices, values),dim=1)
    #     # Convert to homogeneous coordinates
    #     pcd_r = funcs.stack((pcd_r, funcs.ones(pcd_r.shape[0], dtype=funcs.float32)), dim=1)
    #     # Apply the inverse transformation 
    #     T = funcs.mat(self.forward_T[idx], dtype=funcs.float32)
    #     pcd_r = pcd_r @ T.inverse().T  # (N, 4)
    #     # Convert back to 3D coordinates
    #     return pcd_r[:, :3]
    
    @staticmethod
    def compute_4x4_homogeneous_transform(x_min, x_max, y_min, y_max, z_min, z_max, grid_size):
        # Scale and bias for each axis
        a_x = (grid_size - 1) / (x_max - x_min)
        b_x = -x_min * a_x

        a_y = (grid_size - 1) / (y_max - y_min)
        b_y = -y_min * a_y

        a_z = 1.0 / (z_max - z_min)
        b_z = -z_min * a_z

        # Construct the 4x4 transformation matrix
        T = [
            [a_x, 0,   0,   b_x],
            [0,   a_y, 0,   b_y],
            [0,   0,   a_z, b_z],
            [0,   0,   0,   1]
        ]
        return T

    def pcd2img(
        self,
        pcd: Union[np.ndarray, torch.Tensor],
        funcs: MatOps,
    ) -> Union[np.ndarray, torch.Tensor]:
        
        is_torch  = torch.is_tensor(pcd)   
        device    = pcd.device if is_torch else "cpu"

        if pcd is None or len(pcd) == 0:
            gs = 128 if self.grid_size < 0 else self.grid_size
            T = funcs.eye(4, dtype=funcs.float32, device=device)  # Identity transform
            if is_torch:
                depth = funcs.zeros((1, 1, gs, gs), dtype=funcs.float16)
            else:
                depth = funcs.zeros((gs, gs), dtype=funcs.uint8)
            return depth, T
        

        grid_size = int(len(pcd)** 0.5) if self.grid_size < 0 else self.grid_size
        # Use XYZ coordinates only
        xyz = pcd[:, :3]
        # Extract Z (depth) channel
        x = self._mat_funcs[0].copy_mat(xyz[:, 0])
        y = self._mat_funcs[0].copy_mat(xyz[:, 1])
        z = self._mat_funcs[0].copy_mat(xyz[:, 2])

        # Normalize x and y to [0, 1]
        self.x_min = x_min = x.min() if self.x_min==self.inf else self.x_min
        self.x_max = x_max = x.max() if self.x_max==self.inf else self.x_max
        self.y_min = y_min = y.min() if self.y_min==self.inf else self.y_min
        self.y_max = y_max = y.max() if self.y_max==self.inf else self.y_max
        self.z_min = z_min = z.min() if self.z_min==self.inf else self.z_min
        self.z_max = z_max = z.max() if self.z_max==self.inf else self.z_max
        self.z_mean = z_mean = float(z.mean()) if self.z_mean==self.inf else self.z_mean
        # xy_min, xy_max = min(x_min,y_min), max(x_max,y_max)

        if not (x_max - x_min < 1e-5 or y_max - y_min < 1e-5 or z_max - z_min < 1e-5):            
            T = self.compute_4x4_homogeneous_transform(x_min, x_max, y_min, y_max, z_min, z_max, grid_size)
            T = funcs.mat(T, dtype=funcs.float32, device=device)  # (4, 4)

            # Convert pcd to homogeneous
            pcd_hom = funcs.stack([x, y, z, 
                                 funcs.ones(pcd.shape[0], dtype=funcs.float32, device=device)],
                                dim=1)  # (N, 4)
            # Apply transform
            transformed = pcd_hom @ T.T  # (N, 4)

            # Get pixel coordinates and depth
            xi = funcs.clip(funcs.astype_int32(transformed[:, 0]), 0, grid_size - 1)
            yi = funcs.clip(funcs.astype_int32(transformed[:, 1]), 0, grid_size - 1)
            z_norm = transformed[:, 2]
            # z_norm = (z_norm - z_norm.min()) / (z_norm.max() - z_norm.min())  # Normalize to [0, 1]
            z_img = funcs.zeros((grid_size, grid_size), dtype=funcs.float32, device=device)
            z_img[yi, xi] = z_norm
            z_img[z_img == 0] = z_img[z_img > 0].mean()
            z_img = funcs.clip(z_img, 0, 1)  # Clip to [0, 1]

            if not is_torch:
                z_img = funcs.astype_uint8(z_img * 255)
                img_resize = lambda img: cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            else:
                z_img = funcs.astype_float16(z_img.unsqueeze(0).unsqueeze(0))
                img_resize = lambda img: torch.nn.functional.interpolate(img, 
                                size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
                
            if self.img_size:
                z_img = img_resize(z_img)                
                scale = self.img_size / grid_size
                T_rescale = funcs.mat([
                    [scale, 0,     0, 0],
                    [0,     scale, 0, 0],
                    [0,     0,     1, 0],
                    [0,     0,     0, 1],
                ], dtype=funcs.float32, device=device)
                T = T_rescale @ T

        return z_img,T

    def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo] = [], meta={}) -> List[np.ndarray]:
        self._img_data = []
        self.forward_T=[]
        for i, pcd in enumerate(pcds_data):
            z_img,T = self.pcd2img(pcd, self._mat_funcs[i])
            self.forward_T.append(self._mat_funcs[i].to_numpy(T).tolist())
            self._img_data.append(z_img)
        return self._img_data

def filter_inline_points(centerline_points, distance_thresh=2.0):
    filtered_points = {}

    for lbl, dirs in centerline_points.items():
        filtered_points[lbl] = {'x': [], 'y': []}

        for direction in ['x', 'y']:
            points = dirs[direction]
            if len(points) < 2:
                continue  # Not enough points to fit

            # Convert to float32 array for OpenCV
            pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

            # Fit a line: returns (vx, vy, x0, y0)
            line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line.flatten()

            # Compute distances from points to the fitted line
            dx = pts[:, 0, 0] - x0
            dy = pts[:, 0, 1] - y0
            dist = np.abs(vy * dx - vx * dy)  # Perpendicular distance to line

            # Keep points within the distance threshold
            inlier_mask = dist < distance_thresh
            inliers = pts[inlier_mask][:, 0]  # shape (n_inliers, 2)
            inliers = inliers.tolist()  # Convert to list of tuples
            filtered_points[lbl][direction] = inliers

    return filtered_points


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
            # Connected components + stats (fast) — no need to relabel later
            # labels: int32 matrix, stats: [label, x, y, w, h, area]
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels <= 1:
                # No components found, return original binary
                output_image = np.dstack([binary, binary, binary])
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
            # filtered_mask = keep_mask[labels]     # boolean mask of kept components
            # filtered_binary = (filtered_mask * 255).astype(np.uint8)
            output_image = np.dstack([binary, binary, binary])
            # output_image = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)

            centerline_points = {
                lbl: {'x': [], 'y': []}
                for lbl in keep
            }

            for lbl in keep:
                x = stats[lbl, cv2.CC_STAT_LEFT]
                y = stats[lbl, cv2.CC_STAT_TOP]
                w = stats[lbl, cv2.CC_STAT_WIDTH]
                h = stats[lbl, cv2.CC_STAT_HEIGHT]

                region_mask = (labels[y:y+h, x:x+w] == lbl)
                if not region_mask.any():
                    continue

                row_idx = np.arange(h, dtype=np.int64)[:, None]  # (h,1)
                col_idx = np.arange(w, dtype=np.int64)[None, :]  # (1,w)

                # X-direction (per column)
                col_counts = region_mask.sum(axis=0)
                valid_cols = col_counts > 0
                if valid_cols.any() and "_x" in directions[idx]:
                    sum_rows_per_col = (row_idx * region_mask).sum(axis=0)
                    cy = sum_rows_per_col[valid_cols] // col_counts[valid_cols]
                    cx = np.where(valid_cols)[0]
                    points = list(zip(x + cx, y + cy))
                    centerline_points[lbl]['x'].extend(points)

                # Y-direction (per row)
                row_counts = region_mask.sum(axis=1)
                valid_rows = row_counts > 0
                if valid_rows.any() and "_y" in directions[idx]:
                    sum_cols_per_row = (col_idx * region_mask).sum(axis=1)
                    cx2 = sum_cols_per_row[valid_rows] // row_counts[valid_rows]
                    cy2 = np.where(valid_rows)[0]
                    points = list(zip(x + cx2, y + cy2))
                    centerline_points[lbl]['y'].extend(points)
 
            ###############################################################
            # Drawing points:
            filtered_centerline_points = filter_inline_points(centerline_points, distance_thresh=2.0)
            meta[binimg_cls.uuid] = filtered_centerline_points

            for lbl, dirs in filtered_centerline_points.items():
                for px, py in dirs['x']:
                    output_image[int(py), int(px)] = (0, 255, 0)  # GREEN
                for px, py in dirs['y']:
                    output_image[int(py), int(px)] = (0, 0, 255)  # RED

            res.append(output_image)
        
        return res
    binimg_cls._forward_raw=cleanandfit

    ld_back2Pcd = pro3d.processors.Processors.Lambda()
    def back2Pcd(pcds_data, pcds_info, meta):
        res = []
        zi:ZDepthImage = meta.get('ZDepthImage:zi', None)
        z_mean = 1.0#zi.z_mean
        filtered_centerline_points = meta.get(binimg_cls.uuid, {})
        for lbl, dirs in filtered_centerline_points.items():
            for dir in ['x', 'y']:
                if dir not in dirs: continue
                points = dirs[dir]
                if len(points) == 0: continue
                # Convert points to homogeneous coordinates
                points_homogeneous = np.array([[px, py, z_mean, 1] for px, py in points], dtype=np.float32)
                # Apply the inverse transformation
                T = np.linalg.inv(np.array(zi.forward_T[0], dtype=np.float32))
                transformed_points = points_homogeneous @ T.T
                res.append(transformed_points[:, :3])  # Keep only XYZ coordinates

        # to raw pcd coordinates
        T = np.array(meta.get('Lambda:ld_centerz', None).forward_T[0], dtype=np.float32)
        T = np.array(meta.get('PlaneNormalize:pn', None).forward_T[0], dtype=np.float32) @ T
        T = np.linalg.inv(T)
        for i, pcd in enumerate(res):
            pcd_hom = np.stack([pcd[:, 0], pcd[:, 1], pcd[:, 2],
                                np.ones(pcd.shape[0], dtype=np.float32)],
                                axis=1)  # (N, 4)
            transformed = pcd_hom @ T.T  # (N, 4)
            res[i] = transformed[:, :3]  # Keep only XYZ coordinates

        res += pcds_data
        return res
    ld_back2Pcd._forward_raw=back2Pcd

    n_samples=50_000
    voxel_size=0.01
    radius=2.0

    gen = pro3d.generator.NumpyRawFrameFileGenerator(
                            sources=sources,loop=loop,shape_types=[pro3d.ShapeType.XYZ])    
    plane_det = pro3d.processors.Processors.PlaneDetection(distance_threshold=voxel_size*2,alpha=0.1)


    # for testing purposes, we can use a fixed plane
    depth_cv_view      = ImgProcessors.CvImageViewer(title='depth',scale=2.0)
    segdepth_cv_view   = ImgProcessors.CvImageViewer(title='segdepth',scale=2.0)

    mergepcd = pro3d.processors.Processors.MergePCDs()
    pcdviewer = pro3d.processors.Processors.O3DStreamViewer()

    pipes = [
        pro3d.processors.Processors.RandomSample(n_samples=n_samples),
        pro3d.processors.Processors.BackUp(uuid="Backup:rawpcd",device='cpu'),

        #### GPU
        pro3d.processors.Processors.NumpyToTorch(),
        pro3d.processors.Processors.RadiusSelection(radius=radius),
        pro3d.processors.Processors.VoxelDownsample(voxel_size=voxel_size),
        plane_det,
        pro3d.processors.Processors.PlaneNormalize(uuid="PlaneNormalize:pn",
                                detection_uuid=plane_det.uuid,save_results_to_meta=True),
        ld_centerz,
        ld_filter50cm,
        ld_direction,
        pro3d.processors.Processors.BackUp(uuid="Backup:normpcd",device='cpu'),

        ZDepthImage(uuid="ZDepthImage:zi",grid_size=-1,save_results_to_meta=True),
        ImgProcessors.TorchResize(target_size=(224, 224)),
        ImgProcessors.BackUp(uuid="BackUp:depth",device='cpu'),

        ImgProcessors.SegmentationModelsPytorch(ckpt_path='./tmp/epoch=9-step=3950.ckpt',
                #'./tmp/epoch=2-step=948.ckpt',#'./tmp/epoch=183-step=58144.ckpt',epoch=92-step=36735.ckpt
                device='cuda:0',encoder_name='timm-efficientnet-b8',encoder_weights='imagenet'),
        ImgProcessors.TorchGrayToNumpyGray(),
        #### GPU end

        binimg_cls,
        ImgProcessors.BackUp(uuid="BackUp:segdepth",device='cpu'),
    ]

    def run_once(imgs,meta={},pipes=pipes,validate=False):
        try:
            for fn in pipes:
                imgs,meta = (fn.validate if validate else fn)(imgs,meta)
            pcds = meta.get('Backup:rawpcd', None).get_backup_mats()
            pcds,meta = (ld_back2Pcd.validate if validate else ld_back2Pcd)(pcds,meta)

          
            pcds,meta = (mergepcd.validate if validate else mergepcd)(pcds,meta)
            pcds,meta = (pcdviewer.validate if validate else pcdviewer)(pcds,meta)
            imgs = meta.get('BackUp:depth', None).get_backup_mats()
            imgs,meta = (depth_cv_view.validate if validate else depth_cv_view)(imgs,meta)
            imgs = meta.get('BackUp:segdepth', None).get_backup_mats()
            imgs,meta = (segdepth_cv_view.validate if validate else segdepth_cv_view)(imgs,meta)
            

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
                title=f"#### profile ####\n{'  ->  '.join([f.title for f in pipes])}")
        return
    

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
