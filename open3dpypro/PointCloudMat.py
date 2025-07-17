
# Standard Library Imports
import enum
import json
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import uuid

# Third-Party Library Imports
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch
import enum
from typing import Dict, List
from .shmIO import NumpyFloat32SharedMemoryStreamIO

# --- Generalized Device Type ---
class DeviceType(str, enum.Enum):
    CPU = 'cpu'
    GPU = 'gpu'

# --- Generalized Shape Type for Different Modalities ---
class ShapeType(str, enum.Enum):
    XYZ       = 'XYZ'        # (N, 3): positions
    XYZRGB    = 'XYZRGB'     # (N, 6): positions + RGB
    XYZi      = 'XYZi'       # (N, 4): positions + intensity
    XYZiRGB   = 'XYZiRGB'    # (N, 7): positions + intensity + RGB
    XYZRGBi   = 'XYZRGBi'    # (N, 7): positions + RGB + intensity

    XYZN      = 'XYZN'       # (N, 6): positions + normals
    XYZRGBN   = 'XYZRGBN'    # (N, 9): positions + RGB + normals
    XYZiN     = 'XYZiN'      # (N, 7): positions + intensity + normals
    XYZRGBiN  = 'XYZRGBiN'   # (N, 10): positions + RGB + intensity + normals

    def contains_normals(self) -> bool:
        """Check if this shape type includes normals (by suffix 'N')."""
        return self.value.endswith('N')
    def add_normals(self) -> 'ShapeType':
        return ShapeType(self.value+'N')

# --- Color Type (Image-Specific) ---
class ColorDataType(str, enum.Enum):
    Float32Color = 'Float32Color'
    # no supports following
    # Float16Color = 'Float16Color'
    # Uint8Color = 'Uint8Color'
    # JPEG = 'jpeg'
    # UNKNOWN = 'unknown'

# global setting
torch_pcd_dtype = torch.float32
numpy_pcd_dtype = np.float32

class PointCloudMatInfo(BaseModel):
    type: Optional[str] = None
    _dtype: Optional[Union[np.dtype, torch.dtype]] = None
    device: str = ''
    shape_type: Optional[ShapeType] = None
    raw_shape: List[int] = []
    N: int = 0  # Number of points
    uuid: str = ''

    @staticmethod
    def torch_pcd_dtype():
        return torch_pcd_dtype

    @staticmethod
    def numpy_pcd_dtype():
        return numpy_pcd_dtype

    def model_post_init(self, context):
        self.uuid = f'{self.__class__.__name__}:{uuid.uuid4()}'
        return super().model_post_init(context)

    def build(self, pcd_data: Union[np.ndarray, torch.Tensor]):
        # Parse shape_type to Enum
        shape_type = self.shape_type
        self.type = type(pcd_data).__name__

        if isinstance(pcd_data, np.ndarray):
            self._dtype = pcd_data.dtype
            self.device = "cpu"
        elif isinstance(pcd_data, torch.Tensor):
            self._dtype = pcd_data.dtype
            self.device = str(pcd_data.device)
        else:
            raise TypeError(f"pcd_data must be np.ndarray or torch.Tensor, got {type(pcd_data)}")

        if pcd_data.ndim != 2:
            raise ValueError(f"Point cloud data must be 2D (N, D). Got shape: {pcd_data.shape}")

        self.N = pcd_data.shape[0]
        feature_dim = pcd_data.shape[1]

        # Validate feature dimension based on shape type
        expected_dims = {
                ShapeType.XYZ: 3,
                ShapeType.XYZRGB: 6,
                ShapeType.XYZi: 4,
                ShapeType.XYZiRGB: 7,
                ShapeType.XYZRGBi: 7,

                ShapeType.XYZN: 6,
                ShapeType.XYZRGBN: 9,
                ShapeType.XYZiN: 7,
                ShapeType.XYZRGBiN: 10,
            }

        if shape_type not in expected_dims:
            raise ValueError(f"Unsupported shape type {shape_type.value} for point cloud.")

        if feature_dim != expected_dims[shape_type]:
            raise ValueError(
                f"Shape type '{shape_type.value}' expects feature dimension {expected_dims[shape_type]}, "
                f"but got {feature_dim}. Full shape: {pcd_data.shape}"
            )

        self.shape_type = shape_type
        self.raw_shape = list(pcd_data.shape)
        return self

class PointCloudMat(BaseModel):
    shape_type: ShapeType
    info: Optional[PointCloudMatInfo] = None
    _pcd_data: np.ndarray | torch.Tensor = None

    shmIO_mode: Literal[False, 'writer', 'reader'] = False
    shmIO_writer: Optional[NumpyFloat32SharedMemoryStreamIO.Writer] = None
    shmIO_reader: Optional[NumpyFloat32SharedMemoryStreamIO.Reader] = None

    @staticmethod
    def random(shape_type: Union[str, ShapeType], num_points: int = 1000, lib='np', device='cpu'):
        shape_type = ShapeType(shape_type)
        dim_map = {
            ShapeType.XYZ: 3,
            ShapeType.XYZRGB: 6,
            ShapeType.XYZi: 4,
            ShapeType.XYZiRGB: 7,
            ShapeType.XYZRGBi: 7,
        }
        D = dim_map.get(shape_type)
        if D is None:
            raise ValueError(f"Unsupported shape type: {shape_type}")

        if lib == 'np':
            data = np.random.rand(num_points, D).astype(numpy_pcd_dtype)
        elif lib == 'torch':
            data = torch.rand(num_points, D, dtype=torch_pcd_dtype, device=device)
        else:
            raise TypeError(f"Unsupported library: {lib}")

        return PointCloudMat(shape_type=shape_type).build(data)

    def clone(self):
        info_copy = self.info.model_copy()
        data_copy = self._pcd_data.copy() if isinstance(self._pcd_data, np.ndarray) else self._pcd_data.clone()
        return PointCloudMat(shape_type=self.shape_type).build(data_copy, info_copy)

    def zero_clone(self):
        data_copy = self._pcd_data.copy() * 0 if isinstance(self._pcd_data, np.ndarray) else self._pcd_data.clone() * 0
        return PointCloudMat(shape_type=self.shape_type).build(data_copy)

    def random_clone(self):
        num_points, dims = self._pcd_data.shape
        if isinstance(self._pcd_data, np.ndarray):
            data = np.random.rand(num_points, dims).astype(self.info._dtype)
        elif isinstance(self._pcd_data, torch.Tensor):
            data = torch.rand(num_points, dims, dtype=self.info._dtype, device=self.info.device)
        else:
            raise TypeError(f"Unsupported point cloud data type: {type(self._pcd_data)}")
        return PointCloudMat(shape_type=self.shape_type).build(data)

    def model_post_init(self, context):
        if self.shmIO_writer and self.shmIO_reader:
            raise ValueError('PointCloudMat cannot have both shmIO writer and reader.')
        if self.shmIO_writer:
            self.shmIO_writer.build_buffer()
        if self.shmIO_reader:
            self.shmIO_reader.build_buffer()
        return super().model_post_init(context)

    def build(self, pcd_data: Union[np.ndarray, torch.Tensor], info: Optional[PointCloudMatInfo] = None):
        self.info = info or PointCloudMatInfo(shape_type=self.shape_type).build(pcd_data)
        self._pcd_data = pcd_data
        return self

    def build_shmIO(self, shmIO_mode: str = False, target_mat_info: 'PointCloudMat' = None):
        self.shmIO_mode = shmIO_mode
        if self.shmIO_mode == 'writer' and self.info.device == 'cpu':
            self.build_shmIO_writer()
        if self.shmIO_mode == 'reader' and target_mat_info and self.info.device == 'cpu':
            self.build_shmIO_reader(target_mat_info)
        return self

    def build_shmIO_writer(self):
        self.shmIO_mode = 'writer'
        if self.info is None:
            raise ValueError("self.info cannot be None")
        self.shmIO_writer = NumpyFloat32SharedMemoryStreamIO.writer(self.info.uuid, self.info.raw_shape)
        return self

    def build_shmIO_reader(self, target_mat_info: 'PointCloudMat'):
        self.shmIO_mode = 'reader'
        if self.info is None:
            raise ValueError("self.info cannot be None")
        self.shmIO_reader = NumpyFloat32SharedMemoryStreamIO.reader(target_mat_info.uuid, target_mat_info.raw_shape)
        return self

    def to_shmIO_writer(self):
        self.shmIO_mode = 'writer'
        self.shmIO_writer = NumpyFloat32SharedMemoryStreamIO.writer(self.shmIO_reader.shm_name, self.shmIO_reader.array_shape)
        self.shmIO_reader = None

    def to_shmIO_reader(self):
        self.shmIO_mode = 'reader'
        self.shmIO_reader = NumpyFloat32SharedMemoryStreamIO.reader(self.shmIO_writer.shm_name, self.shmIO_writer.array_shape)
        self.shmIO_writer = None

    def release(self):
        if self.shmIO_reader:
            self.shmIO_reader.close()
        if self.shmIO_writer:
            self.shmIO_writer.close()

    def copy(self) -> 'PointCloudMat':
        if isinstance(self._pcd_data, np.ndarray):
            return PointCloudMat(shape_type=self.info.shape_type).build(self._pcd_data.copy())
        elif isinstance(self._pcd_data, torch.Tensor):
            return PointCloudMat(shape_type=self.info.shape_type).build(self._pcd_data.clone())
        else:
            raise TypeError("pcd_data must be np.ndarray or torch.Tensor")

    def update_mat(self, pcd_data: Union[np.ndarray, torch.Tensor]) -> 'PointCloudMat':
        self.info = PointCloudMatInfo().build(pcd_data, shape_type=self.info.shape_type)
        self.unsafe_update_mat(pcd_data)
        return self

    def unsafe_update_mat(self, pcd_data: Union[np.ndarray, torch.Tensor]) -> 'PointCloudMat':
        if self.shmIO_mode:
            self.shmIO_writer.write(pcd_data)
        self._pcd_data = pcd_data
        return self

    def data(self) -> Union[np.ndarray, torch.Tensor]:
        if self.shmIO_mode:
            return self.shmIO_reader.read()
        return self._pcd_data

    # --- Type and Shape Requirement Methods ---
    def is_ndarray(self):
        return isinstance(self._pcd_data, np.ndarray)

    def is_torch_tensor(self):
        return isinstance(self._pcd_data, torch.Tensor)

    def require_ndarray(self):
        if not isinstance(self._pcd_data, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(self._pcd_data)}")

    def require_torch_tensor(self):
        if not isinstance(self._pcd_data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(self._pcd_data)}")

    def require_torch_float(self):
        self.require_torch_tensor()
        if self._pcd_data.dtype != torch_pcd_dtype:
            raise TypeError(f"Point cloud data must be {torch_pcd_dtype}. Got {self._pcd_data.dtype}")

    def require_shape_type(self, shape_type: ShapeType):
        if self.info.shape_type != shape_type:
            raise TypeError(f"Expected shape type {shape_type.value}, got {self.info.shape_type.value}")

    def require_shape_types(self, shape_types: list[ShapeType]):
        if self.info.shape_type in shape_types:
            raise TypeError(f"Expected shape types {shape_types}, got {self.info.shape_type.value}")
       
class PointCloudMatProcessor(BaseModel):
    class MetaData(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

    title: str
    uuid: str = ''

    save_results_to_meta: bool = False
    input_mats: List[PointCloudMat] = []
    out_mats: List[PointCloudMat] = []
    meta: dict = {}

    num_devices: List[str] = ['cpu']
    num_gpus: int = 0
    _enable: bool = True

    forward_T:List[ List[List[float]] ]=[]

    _mat:Callable = lambda pylist,dtype,device=None: NotImplementedError()
    _eye:Callable = lambda size,dtype,device=None: NotImplementedError()
    _ones:Callable = lambda shape,dtype,device=None: NotImplementedError()
    _hstack:Callable = lambda arrays: NotImplementedError()
    _norm:Callable = lambda x: NotImplementedError()
    _dot:Callable = lambda a,b: NotImplementedError()
    _cross:Callable = lambda a,b: NotImplementedError()
    _matmul:Callable = lambda a,b: NotImplementedError()
    _to_numpy:Callable = lambda x: NotImplementedError()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def init_common_utility_methods(self,pcd:PointCloudMat):
        if pcd.is_ndarray():
            self._mat = lambda pylist,dtype,device=None: np.array(pylist,dtype=dtype)
            self._eye = lambda size,dtype,device=None: np.eye(size, dtype=dtype)
            self._ones = lambda shape,dtype,device=None: np.ones(shape, dtype=dtype)
            self._hstack = lambda arrays: np.hstack(arrays)
            self._norm = lambda x: np.linalg.norm(x)
            self._dot = lambda a,b: np.dot(a, b)
            self._cross = lambda a,b: np.cross(a, b)
            self._matmul = lambda a,b: a @ b
            self._to_numpy = lambda x: x
        if pcd.is_torch_tensor():
            self._mat = lambda pylist,dtype,device=None: torch.tensor(pylist,dtype=dtype,device=device)
            self._eye = lambda size,dtype,device=None: torch.eye(size, dtype=dtype, device=device)
            self._ones = lambda shape,dtype,device=None: torch.ones(shape, dtype=dtype, device=device)
            self._hstack = lambda arrays: torch.cat(arrays, dim=1)
            self._norm = lambda x: torch.norm(x)
            self._dot = lambda a,b: torch.dot(a, b)
            self._cross = lambda a,b: torch.cross(a, b)
            self._matmul = lambda a,b: torch.matmul(a, b)
            self._to_numpy = lambda x: x.cpu().numpy()

    def print(self, *args):
        print(f'##############[{self.uuid}]#################')
        print(f'[{self.uuid}]', *args)
        print(f'############################################')

    def model_post_init(self, context: Any, /) -> None:
        if not self.title:
            self.title = self.__class__.__name__
        if not self.uuid:
            self.uuid = f'{self.__class__.__name__}:{uuid.uuid4()}'
        return super().model_post_init(context)

    def is_enable(self):
        return self._enable

    def on(self):
        self._enable = True

    def off(self):
        self._enable = False

    def devices_info(self, gpu=True, multi_gpu=-1):
        self.num_devices = ['cpu']
        self.num_gpus = 0
        if gpu and torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            if multi_gpu <= 0 or multi_gpu > self.num_gpus:
                self.num_devices = [f"cuda:{i}" for i in range(self.num_gpus)]
            else:
                self.num_devices = [f"cuda:{i}" for i in range(multi_gpu)]
        return self.num_devices

    def validate_pcd(self, idx: int, pcd: PointCloudMat):
        """
        Implement per-point-cloud validation logic.
        Example: pcd.require_ndarray(), pcd.require_shape_type(ShapeType.XYZ), etc.
        """
        raise NotImplementedError()

    def validate(self, pcds: List[PointCloudMat], meta: Dict = {}, run=True):
        self.input_mats = pcds
        input_mats = [None] * len(pcds)
        for i, pcd in enumerate(pcds):
            self.validate_pcd(i, pcd)
            if pcd.shmIO_mode == False:
                input_mats[i] = pcd
            elif pcd.shmIO_mode == 'writer':
                pcd = pcd.model_copy()
                pcd.to_shmIO_reader()
            else:
                raise ValueError(f"pcd shmIO_mode must be False or writer. Got {pcd.shmIO_mode}")
            input_mats[i] = pcd

        self.input_mats = input_mats
        self.forward_T  = [np.eye(4).tolist() for i in self.input_mats]
        if run:
            return self(self.input_mats, meta)
        return self.input_mats

    def build_out_mats(self, validated_pcds: List[PointCloudMat], converted_raw_pcds):
        self.out_mats = [
            PointCloudMat(shape_type=old.info.shape_type).build(pcd)
            for old, pcd in zip(validated_pcds, converted_raw_pcds)
        ]
        return self.out_mats

    def forward_raw(self, pcds: List[Any], pcd_infos: List[PointCloudMatInfo] = [], meta={}) -> List[Any]:
        """
        To be implemented by subclass.
        pcds: raw np.ndarray or torch.Tensor data list.
        pcd_infos: corresponding PointCloudMatInfo list.
        """
        raise NotImplementedError()

    def forward(self, pcds: List[PointCloudMat], meta: Dict) -> Tuple[List[PointCloudMat], Dict]:
        if pcds and pcds[0].shmIO_mode == False:
            pcds = pcds
        elif self.input_mats:
            pcds = self.input_mats

        input_infos = [pcd.info for pcd in pcds]
        forwarded_pcds = [pcd.data() for pcd in pcds]

        if self._enable:
            forwarded_pcds = self.forward_raw(forwarded_pcds, input_infos, meta)

        if len(self.out_mats) == len(forwarded_pcds):
            output_pcds = [
                self.out_mats[i].unsafe_update_mat(forwarded_pcds[i])
                for i in range(len(forwarded_pcds))
            ]
        else:
            output_pcds = self.build_out_mats(self.input_mats, forwarded_pcds)
            shmIO_mode = any([i.shmIO_mode for i in pcds])
            if shmIO_mode:
                for o in output_pcds:
                    o.shmIO_mode = 'writer'
                    o.build_shmIO_writer()

        self.out_mats = output_pcds

        if self.save_results_to_meta:
            meta[self.uuid] = self

        return output_pcds, meta

    def __call__(self, pcds: List[PointCloudMat], meta: dict = {}):
        return self.forward(pcds, meta)

    def release(self):
        for i in self.input_mats:
            i.release()
        for i in self.out_mats:
            i.release()

    def __del__(self):
        self.release()