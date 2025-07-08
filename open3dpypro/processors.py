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
from typing import Callable, List, Optional

# ===============================
# Third-Party Library Imports
# ===============================
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

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
        
    class CPUNormals(PointCloudMatProcessor):
        title:str='calc_normals'
        input_shape_types: list['ShapeType'] = []
        
        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            pcd.require_ndarray()
            self.input_shape_types.append(pcd.shape_type)

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            for i,pcd in enumerate(pcds_data):
                if not self.input_shape_types[i].contains_normals():
                    ns = PointCloudBase(pcd[:,:3]).estimate_normals().get_normals()
                    pcd = np.hstack([pcd,ns])
            return pcds_data

    class RandomSample(PointCloudMatProcessor):
        title:str='rand_sample'
        n_samples:int = 1000
        copy_pcd:bool = False

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            if pcd.is_ndarray():
                pcd.require_ndarray()
            if pcd.is_torch_tensor():
                pcd.require_torch_float()

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            res = []
            for i,pcd in enumerate(pcds_data):
                idx = np.random.randint(0,len(pcd),(self.n_samples))
                pcd = pcd[idx]
                if self.copy_pcd:
                    if isinstance(pcd, np.ndarray):
                        pcd = pcd.copy()                    
                    if isinstance(pcd, torch.Tensor):
                        pcd = pcd.clone()
                res.append(pcd)
            return res

    class PlaneDetection(PointCloudMatProcessor):
        title:str='plane_detection'
        distance_threshold: float = 0.01
        arange: bool = False
        voxel_down_sample:float=0.01

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            pcd.require_ndarray()

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            res = []
            for i,pcd in enumerate(pcds_data):
                if not self.arange:
                    z_mean = pcd[:,2].mean()
                    z_min = pcd[:,2].min()
                    
                    lower,upper = z_mean,z_min
                else:
                    lower,upper = self.arange

                if lower>upper:
                    lower,upper = upper,lower
                pch = PointCloud(pcd)
                pch = pch.select_by_bool( np.logical_and(lower<pcd[:,2],pcd[:,2]<upper) )
                pch = pch.voxel_down_sample(self.voxel_down_sample)
                model,planeidx = pch.segment_plane(
                    thickness=self.distance_threshold,ransac_n=3,num_iterations=450)
                res.append(model)
            meta[self.uuid]=res
            return pcds_data

    class PlaneNormalize(PointCloudMatProcessor):
        title:str='plane_normalize'
        detection_uuid:str

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            pcd.require_ndarray()

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            models = meta[self.detection_uuid]
            res = []
            for i,pcd in enumerate(pcds_data):
                model = models[i]
                pch,T = PointCloud(pcd).rotate_to_plane(model)
                self.forward_T[i] = T.tolist()
                res.append(pch.get_points())
            return res

    class Lambda(PointCloudMatProcessor):
        title:str='lambda'
        _forward_raw:Callable = None

        def validate_pcd(self, pcd_idx, pcd:PointCloudMat):
            pcd.require_ndarray()

        def forward_raw(self, pcds_data: List[np.ndarray], pcds_info: List[PointCloudMatInfo]=[], meta={}) -> List[np.ndarray]:
            models = meta[self.detection_uuid]
            res = []
            for i,pcd in enumerate(pcds_data):
                model = models[i]
                pch,T = PointCloud(pcd).rotate_to_plane(model)
                self.forward_T[i] = T.tolist()
                res.append(pch.get_points())
            return res
        
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
                    print(f"Skipping point cloud {i}: flat x or y range.")
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