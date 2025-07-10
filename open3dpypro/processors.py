# ===============================
# Standard Library Imports
# ===============================
from collections import defaultdict
from datetime import datetime
import enum
import json
import math
from multiprocessing import shared_memory
import multiprocessing
from typing import Any, Callable, List, Optional

# ===============================
# Third-Party Library Imports
# ===============================
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
import open3d as o3d
import numpy as np
from typing import List
# ===============================
# Ultralytics YOLO Imports
# ===============================
# from ultralytics import YOLO
# from ultralytics.utils import ops

# ===============================
# TensorRT & CUDA Imports
# ===============================
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

# ===============================
# Custom Modules
# ===============================
from .PointCloudMat import PointCloudMatInfo, PointCloudMatProcessor, PointCloudMat, ShapeType
from .PointCloud import PointCloud, PointCloudBase, PointCloudSelections
from .shmIO import NumpyFloat32SharedMemoryStreamIO

logger = print

class Processors:
    class DoingNothing(PointCloudMatProcessor):
        title:str='doing_nothing'
        def validate_pcd(self, pcd_idx, pcd):
            pass
        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            return pcds_data
        
    class NumpyToTorch(PointCloudMatProcessor):
        title: str = 'numpy_to_torch'

        def model_post_init(self, context):
            self.devices_info(gpu=True, multi_gpu=-1)            
            return super().model_post_init(context)
        
        def validate_pcd(self, pcd_idx, pcd: PointCloudMat):
            pcd.require_ndarray()

        def forward_raw(
            self,
            pcds_data: List[np.ndarray],
            pcds_info: Optional[List[PointCloudMatInfo]] = None,
            meta: Optional[dict] = None
        ) -> List[torch.Tensor]:
            res = []
            for i,pcd in enumerate(pcds_data):
                tensor_pcd = torch.from_numpy(pcd).type(
                                    PointCloudMatInfo.torch_pcd_dtype()
                                ).to(self.num_devices[i%self.num_gpus])
                res.append(tensor_pcd)
            return res
        
    class TorchToNumpy(PointCloudMatProcessor):
        title: str = 'torch_to_numpy'

        def validate_pcd(self, pcd_idx, pcd: PointCloudMat):
            pcd.require_torch_float()

        def forward_raw(
            self,
            pcds_data: List[torch.Tensor],
            pcds_info: Optional[List[PointCloudMatInfo]] = None,
            meta: Optional[dict] = None
        ) -> List[np.ndarray]:
            res = []
            for i,pcd in enumerate(pcds_data):
                res.append(pcd.cpu().numpy())
            return res
        
    class CPUNormals(PointCloudMatProcessor):
        title:str='calc_normals'
        input_shape_types: list['ShapeType'] = []
        
        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            pcd.require_ndarray()
            self.input_shape_types.append(pcd.shape_type)
        
        def build_out_mats(self, validated_pcds: List[PointCloudMat], converted_raw_pcds):
            self.out_mats = [
                PointCloudMat(shape_type=old.info.shape_type.add_normals()).build(pcd)
                for old, pcd in zip(validated_pcds, converted_raw_pcds)
            ]
            return self.out_mats

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            res = []
            for i,pcd in enumerate(pcds_data):
                if not self.input_shape_types[i].contains_normals():
                    ns = PointCloudBase(pcd[:,:3]).estimate_normals().get_normals()
                    res.append(np.hstack([pcd,ns]))
            return res

    class RandomSample(PointCloudMatProcessor):
        title:str='rand_sample'
        n_samples:int = 1000
        copy_pcd:bool = False
        _models:list = []

        def build(self):            
            for i,pcd in enumerate(self.input_mats):
                                
                if pcd.info.device=='cpu':
                    def cpu_model(pcd:np.ndarray,n_samples=self.n_samples,copy_pcd=self.copy_pcd):
                        if len(pcd)>n_samples:
                            idx = np.random.randint(0,len(pcd),(n_samples,))
                            pcd = pcd[idx]
                            if copy_pcd: return pcd.copy()
                        return pcd
                    self._models.append(cpu_model)

                elif 'cuda' in pcd.info.device:                
                    def cuda_model(pcd:torch.Tensor,device=self.num_devices[i%self.num_gpus],
                                   n_samples=self.n_samples,copy_pcd=self.copy_pcd):
                        if len(pcd)>n_samples:
                            idx = torch.randint(0,len(pcd),(n_samples,),device=device)
                            pcd = pcd[idx]
                            if copy_pcd: return pcd.clone()
                        return pcd
                    self._models.append(cuda_model)
                
                else:
                    raise ValueError('not support')

        def model_post_init(self, context):
            self.build()
            return super().model_post_init(context)

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            if pcd.is_ndarray():
                pcd.require_ndarray()
                self.build()
            if pcd.is_torch_tensor():
                pcd.require_torch_float()
                self.devices_info()
                self.build()

        def forward_raw(self, pcds_data: List[np.ndarray|torch.Tensor], 
                        pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray|torch.Tensor]:
            res = []
            for i,pcd in enumerate(pcds_data):
                model = self._models[i]
                res.append(model(pcd))
            return res
        
    class RadiusSelection(PointCloudMatProcessor):
        title:str='radius_selection'
        radius:float = 5.0
        _models:list = []

        def build(self):            
            for i,pcd in enumerate(self.input_mats):
                                
                if pcd.info.device=='cpu':
                    def cpu_model(pcd:np.ndarray,r=self.radius):
                        pch = PointCloud(pcd[:,:3]).select_by_radius(r=r)
                        return pch.get_points()
                    self._models.append(cpu_model)

                elif 'cuda' in pcd.info.device:                
                    def cuda_model(pcd:torch.Tensor,r=self.radius):
                        xyz = pcd[:, :3]
                        mask = (xyz.norm(dim=1) <= r)
                        return pcd[mask]

                    self._models.append(cuda_model)
                
                else:
                    raise ValueError('not support')

        def model_post_init(self, context):
            self.build()
            return super().model_post_init(context)

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            if pcd.is_ndarray():
                pcd.require_ndarray()
                self.build()
            if pcd.is_torch_tensor():
                pcd.require_torch_float()
                self.devices_info()
                self.build()

        def forward_raw(self, pcds_data: List[np.ndarray|torch.Tensor], 
                        pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray|torch.Tensor]:
            res = []
            for i,pcd in enumerate(pcds_data):
                model = self._models[i]
                res.append(model(pcd))
            return res
        
    class VoxelDownsample(PointCloudMatProcessor):
        title:str='voxel_sample'
        voxel_size:float=0.1
        _models:list = []

        def build(self):            
            for i,pcd in enumerate(self.input_mats):
                                
                if pcd.info.device=='cpu':
                    def cpu_model(pcd:np.ndarray,voxel_size=self.voxel_size):
                        pch,idxmat,vec = PointCloud(pcd[:,:3]).voxel_down_sample_and_trace(voxel_size)
                        return pcd[idxmat.max(1)]
                    self._models.append(cpu_model)

                elif 'cuda' in pcd.info.device:                
                    def cuda_model(pcd:torch.Tensor,device=self.num_devices[i%self.num_gpus],
                                   voxel_size=self.voxel_size):
                        # Compute voxel indices
                        voxel_indices = torch.floor(pcd / voxel_size).int()
                        # Create unique keys for voxels
                        keys = voxel_indices[:, 0] * 73856093 + voxel_indices[:, 1] * 19349663 + voxel_indices[:, 2] * 83492791
                        unique_keys, inverse_indices = torch.unique(keys, return_inverse=True)
                        # Sort inverse_indices so that same voxels are together
                        _, sorted_indices = torch.sort(inverse_indices)
                        sorted_inverse = inverse_indices[sorted_indices]
                        # Get mask of first occurrence in each group
                        mask = torch.ones_like(sorted_inverse, dtype=torch.bool)
                        mask[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
                        # First indices per voxel
                        first_indices = sorted_indices[mask]
                        return pcd[first_indices]
                    
                    self._models.append(cuda_model)
                
                else:
                    raise ValueError('not support')

        def model_post_init(self, context):
            self.build()
            return super().model_post_init(context)

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            if pcd.is_ndarray():
                pcd.require_ndarray()
                self.build()
            if pcd.is_torch_tensor():
                pcd.require_torch_float()
                self.devices_info()
                self.build()

        def forward_raw(self, pcds_data: List[np.ndarray|torch.Tensor], 
                        pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray|torch.Tensor]:
            res = []
            for i,pcd in enumerate(pcds_data):
                model = self._models[i]
                res.append(model(pcd))
            return res
        
        # def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
        #     pcd.require_ndarray()

        # def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
        #     res = []
        #     for i,pcd in enumerate(pcds_data):
        #         pch,idxmat,vec = PointCloud(pcd[:,:3]).voxel_down_sample_and_trace(self.voxel_size)
        #         res.append(pcd[idxmat.max(1)])
        #     return res

    class RemoveStatisticalOutlier(PointCloudMatProcessor):
        title:str='remove_statistical_outlier'
        nb_neighbors: int = 20
        std_ratio: float = 2.0
        print_progress: bool = False
        
        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            pcd.require_ndarray()

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            res = []
            for i,pcd in enumerate(pcds_data):
                pch,idxmat = PointCloud(pcd[:,:3]).remove_statistical_outlier(self.nb_neighbors,self.std_ratio)
                res.append(pcd[idxmat])
            return res
        
    class PlaneDetection(PointCloudMatProcessor):
        title:str='plane_detection'
        distance_threshold: float = 0.01
        arange: bool = False
        best_planes:list[list[float]] = []
        alpha:float = 0.0
        num_iterations:int = 512
        num_iteration_batch:int = 256
        _models:list = []

        def ransac_plane_detection_torch(self,
            points: torch.Tensor,
            distance_threshold: float = 0.01,
            num_iterations: int = 512
        ):
            device = points.device

            best_inlier_count = 0
            best_plane = None
            best_inlier_mask = None

            for _ in range(num_iterations):
                # Randomly sample 3 points
                idx = torch.randint(0, points.shape[0], (3,), device=device)
                p1, p2, p3 = points[idx]

                # Compute normal vector
                v1 = p2 - p1
                v2 = p3 - p1
                normal = torch.cross(v1, v2)

                if torch.norm(normal) < 1e-6:
                    continue  # Degenerate, skip

                normal = normal / torch.norm(normal)

                d = -torch.dot(normal, p1)
                
                if normal[2] < 0:
                    normal = -normal
                    d = -d

                # Compute distances to plane
                distances = torch.abs(torch.matmul(points, normal) + d)

                # Find inliers
                inlier_mask = distances < distance_threshold
                inlier_count = inlier_mask.sum().item()

                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    # Format plane as [a, b, c, d]
                    best_plane = torch.cat([normal, d.unsqueeze(0)]).detach().cpu().numpy()
                    best_inlier_mask = inlier_mask
                    
            return best_plane, best_inlier_mask
                
        def ransac_plane_detection_torch_batched(self,
            points: torch.Tensor,
            distance_threshold: float = 0.01,
            num_iterations: int = 512,
            batch_size: int = 256,
        ):
            """
            Batched RANSAC plane detection on GPU using PyTorch, always ensuring normals point +Z.

            Args:
                points (torch.Tensor): (N, 3).
                distance_threshold (float): Inlier distance threshold.
                num_iterations (int): Number of outer iterations.
                batch_size (int): Number of plane hypotheses per batch.

            Returns:
                best_plane (np.ndarray): [a, b, c, d].
                best_inlier_mask (torch.Tensor): Boolean mask (N,).
            """
            device = points.device
            num_points = points.shape[0]

            best_inlier_count = 0
            best_plane = None
            best_inlier_mask = None

            for _ in range(num_iterations//batch_size):
                # Sample batch_size * 3 indices
                idx = torch.randint(0, num_points, (batch_size, 3), device=device)
                p1 = points[idx[:, 0]]
                p2 = points[idx[:, 1]]
                p3 = points[idx[:, 2]]

                v1 = p2 - p1
                v2 = p3 - p1
                normals = torch.cross(v1, v2)

                norms = torch.norm(normals, dim=1, keepdim=True)
                valid_mask = norms[:, 0] > 1e-6

                normals = normals / norms.clamp(min=1e-6)
                ds = -torch.einsum("bi,bi->b", normals, p1)

                # Flip normals toward sensor center
                dot_to_center = ds
                flip_mask = dot_to_center < 0
                normals[flip_mask] = -normals[flip_mask]
                ds[flip_mask] = -ds[flip_mask]

                # Evaluate distances: shape (N, batch_size)
                dists = torch.abs(torch.matmul(points, normals.T) + ds)

                # Compute inliers
                inlier_masks = dists < distance_threshold
                inlier_counts = inlier_masks.sum(dim=0)

                # Find best in this batch
                batch_best_count = inlier_counts.max().item()
                batch_best_idx = inlier_counts.argmax()

                if batch_best_count > best_inlier_count:
                    best_inlier_count = batch_best_count
                    best_n = normals[batch_best_idx]
                    best_d = ds[batch_best_idx]
                    best_plane = torch.cat([best_n, best_d.unsqueeze(0)]).detach().cpu().numpy()
                    best_inlier_mask = inlier_masks[:, batch_best_idx]
            return best_plane, best_inlier_mask
        
        def build(self):            
            for i,pcd in enumerate(self.input_mats):
                                
                if pcd.info.device=='cpu':
                    def cpu_model(pcd:np.ndarray,lower=None,upper=None):
                        pch = PointCloud(pcd[:,:3])
                        if lower is not None and upper is not None:
                            pch = pch.select_by_bool( np.logical_and(lower<pcd[:,2],pcd[:,2]<upper) )
                        plane_model,_ = pch.segment_plane(
                            thickness=self.distance_threshold,ransac_n=3,num_iterations=self.num_iterations)
                        
                        plane_model = np.asarray(plane_model)
                        n = plane_model[:3]
                        d = plane_model[3]
                        # Compute point on plane
                        point_on_plane = -d * n
                        # Vector to sensor center
                        v = np.zeros(3) - point_on_plane
                        # Flip if needed
                        if np.dot(n, v) < 0:
                            plane_model = -plane_model
                        return plane_model
                    self._models.append(cpu_model)

                elif 'cuda' in pcd.info.device:                
                    def cuda_model(pcd:torch.Tensor,lower=None,upper=None,self=self):
                        pcd = pcd[:,:3]
                        if lower is not None and upper is not None:
                            pcd = pcd[torch.logical_and(lower<pcd[:,2],pcd[:,2]<upper)]
                        best_plane,_ = self.ransac_plane_detection_torch_batched(
                                                        pcd,self.distance_threshold,
                                                        self.num_iterations,self.num_iteration_batch)                    
                        return best_plane
                    self._models.append(cuda_model)
                
                else:
                    raise ValueError('not support')
                
        def model_post_init(self, context):
            self.build()
            return super().model_post_init(context)

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            if pcd.is_ndarray():
                pcd.require_ndarray()
                self.build()
            if pcd.is_torch_tensor():
                pcd.require_torch_float()
                self.devices_info()
                self.build()
            self.best_planes.append([0.,0.,0.,0.])

        def forward_raw(self, pcds_data: List[np.ndarray|torch.Tensor], 
                        pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray|torch.Tensor]:
            for i,pcd in enumerate(pcds_data):
                model = self._models[i]
                plane = model(pcd)
                self.best_planes[i] = (np.asarray(self.best_planes[i])*(1.0-self.alpha)+plane*self.alpha).tolist()
            meta[self.uuid]=self.best_planes
            return pcds_data
        
    class PlaneNormalize(PointCloudMatProcessor):
        title:str='plane_normalize'
        detection_uuid:str
        filter_pcd:bool=False

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            pcd.require_ndarray()

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            models = meta[self.detection_uuid]
            res = []
            for i,pcd in enumerate(pcds_data):
                model = models[i]
                pch,T = PointCloud(pcd[:,:3]).rotate_to_plane(model)
                self.forward_T[i] = T.tolist()
                res.append(np.hstack([pch.get_points(),pcd[:,3:]]))
            return res

    class Lambda(PointCloudMatProcessor):
        title:str='lambda'
        _forward_raw:Callable = lambda pcds_data, pcds_info, meta:None

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            pcd.require_ndarray()

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            return self._forward_raw(pcds_data,pcds_info,meta)
        
    class ZDepthViewer(PointCloudMatProcessor):
        title: str = 'z_depth_viewer'
        bg:tuple[int,int,int] = (125,125,125)
        grid_size: int = 256  # Grid resolution (e.g., 256 x 256)

        def validate_pcd(self, pcd_idx, pcd: PointCloudMat):
            if pcd.is_ndarray():
                pcd.require_ndarray()
            if pcd.is_torch_tensor():
                pcd.require_torch_float()

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo] = [], meta={}) -> List[np.ndarray]:
            for i, pcd in enumerate(pcds_data):
                z_img_color = np.zeros((self.grid_size,self.grid_size,3),dtype=np.uint8)

                if len(pcd)>0:
                    # Use XYZ coordinates only
                    xyz = pcd[:, :3]
                    # Extract Z (depth) channel
                    x = xyz[:, 0]
                    y = xyz[:, 1]
                    z = xyz[:, 2]

                    # Normalize x and y to [0, 1]
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()

                    if x_max - x_min < 1e-5 or y_max - y_min < 1e-5:
                        logger(f"Skipping point cloud {i}: flat x or y range.")
                        continue

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
                        z_min = np.nanmin(z_grid)
                        z_grid = np.where(np.isnan(z_grid), z_min, z_grid)

                    # Normalize z to [0, 255]
                    z_min, z_max = z_grid.min(), z_grid.max()
                    if z_max - z_min < 1e-5:
                        z_norm = np.zeros_like(z_grid)
                    else:
                        z_norm = (z_grid - z_min) / (z_max - z_min)

                    z_img = (z_norm * 255).astype(np.uint8)
                    # Optional: apply colormap
                    z_img_color = cv2.applyColorMap(z_img, cv2.COLORMAP_JET)
                    z_img_color[z_img==0] = self.bg

                # Show image
                cv2.imshow(f"Z Depth Viewer {i}", z_img_color)
                cv2.waitKey(1)

            return pcds_data

        def release(self):
            for i,m in enumerate(self.input_mats):
                cv2.destroyWindow(f"Z Depth Viewer {i}")
            return super().release()

    class O3DStreamViewer(PointCloudMatProcessor):
        title: str = "o3d_stream_viewer"
        axis_size: float = 0.5
        _vis:Any = None
        _axis:Any = None        
        _pchs:list[PointCloud] = []

        def model_post_init(self, context):
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window(window_name="Open3D Stream Viewer")
            self._axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_size)
            self._vis.add_geometry(self._axis)
            return super().model_post_init(context)

        def validate_pcd(self, pcd_idx, pcd: PointCloudMat):
            pcd.require_ndarray()
            self._pchs.append(PointCloud(pcd.data()[:, :3]))
            self._vis.add_geometry(self._pchs[-1].pcd)

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo] = [], meta={}) -> List[np.ndarray]:
            pchs = []
            for i, pcd in enumerate(pcds_data):
                pch = self._pchs[i]
                pch.set_points(pcd[:, :3])
                # Optionally, add colors if available
                # if pcd.shape[1] >= 6:
                #     colors = pcd[:, 3:6]
                #     colors = np.clip(colors, 0, 1) if colors.max() <= 1 else np.clip(colors / 255.0, 0, 1)
                #     pch.colors = o3d.utility.Vector3dVector(colors)
                # else:
                #     pch.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (xyz.shape[0], 1)))
                self._vis.update_geometry(pch.pcd)

            self._vis.poll_events()
            self._vis.update_renderer()

            return pcds_data

        def release(self):
            self._vis.destroy_window()
            return super().release()

class PointCloudMatProcessors(BaseModel):
    @staticmethod    
    def dumps(pipes:list[PointCloudMatProcessor]):
        return json.dumps([p.model_dump() for p in pipes])
    
    @staticmethod
    def loads(pipes_json:str)->list[PointCloudMatProcessor]:
        processors = {k: v for k, v in Processors.__dict__.items() if '__' not in k}
        return [processors[f'{p["uuid"].split(":")[0]}'](**p) 
                for p in json.loads(pipes_json)]

    @staticmethod    
    def run_once(imgs,meta={},
            pipes:list['PointCloudMatProcessor']=[],
            validate=False):
        try:
            for fn in pipes:
                imgs,meta = (fn.validate if validate else fn)(imgs,meta)
        except Exception as e:
            logger(fn.uuid,e)
            raise e
        return imgs,meta
        
    @staticmethod    
    def run(gen,
            pipes:list['PointCloudMatProcessor']=[],
            meta = {},validate_once=False):
        if isinstance(pipes, str):
            pipes = PointCloudMatProcessors.loads(pipes)
        for imgs in gen:
            PointCloudMatProcessors.run_once(imgs,meta,pipes,validate_once)
            if validate_once:return

    @staticmethod    
    def validate_once(gen,
            pipes:list['PointCloudMatProcessor']=[]):
        PointCloudMatProcessors.run(gen,pipes,validate_once=True)

    @staticmethod
    def worker(pipes_serialized):
        pipes = PointCloudMatProcessors.loads(pipes_serialized)
        imgs,meta = [],{}
        while True:
            for fn in pipes:
                imgs,meta = fn(imgs,meta)

    @staticmethod
    def run_async(pipes: list[PointCloudMatProcessor] | str):
        if isinstance(pipes, str):
            pipes_serialized = pipes
        else:
            pipes_serialized = PointCloudMatProcessors.dumps(pipes)
            
        p = multiprocessing.Process(target=PointCloudMatProcessors.worker, args=(pipes_serialized,))
        p.start()
        return p