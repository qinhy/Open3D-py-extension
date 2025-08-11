import json
import multiprocessing
import time
import uuid
from typing import Iterator, List, Literal

import numpy as np
from pydantic import BaseModel
import queue

from .PointCloudMat import ShapeType, PointCloudMat

logger = print

class PointCloudMatGenerator(BaseModel):
    sources: List[str]
    shape_types: List[ShapeType]
    uuid: str = ''
    shmIO_mode: Literal[False, 'writer', 'reader'] = False
    fps: int = -1
    _min_frame_time: float = 0.0

    _resources: List = []
    _frame_generators: List = []
    output_mats: List[PointCloudMat] = []

    def model_post_init(self, context):
        self._min_frame_time = 1.0 / self.fps if self.fps > 0 else 0
        self.uuid = f'{self.__class__.__name__}:{uuid.uuid4()}'

        if not self.sources:
            raise ValueError("Empty sources.")
        if not self.shape_types:
            raise ValueError("Empty shape_types.")

        self._frame_generators = [self.create_frame_generator(i, src) for i, src in enumerate(self.sources)]

        if not self._frame_generators:
            raise ValueError("Empty frame_generators.")

        if not self.output_mats:
            self.output_mats = []
            for i,gen in enumerate(self._frame_generators):
                outmat = PointCloudMat(shape_type=self.shape_types[i]).build(next(gen))
                self.output_mats.append(outmat)

        for mat in self.output_mats:
            mat.shmIO_mode = self.shmIO_mode
            if mat.shmIO_writer:
                mat.shmIO_writer.build_buffer()
            elif mat.shmIO_mode == 'writer':
                mat.build_shmIO_writer()

        return super().model_post_init(context)

    def register_resource(self, resource):
        self._resources.append(resource)
        return resource

    @staticmethod
    def has_func(obj, name):
        return callable(getattr(obj, name, None))

    def release_resources(self):
        cleanup_methods = [
            "exit", "end", "teardown",
            "stop", "shutdown", "terminate",
            "join", "cleanup", "deactivate",
            "release", "close", "disconnect",
            "destroy",
        ]
        for res in self._resources:
            for method in cleanup_methods:
                if self.has_func(res, method):
                    try:
                        getattr(res, method)()
                    except Exception as e:
                        logger(f"Error during {method} on {res}: {e}")
        self._resources.clear()

    def create_frame_generator(self, idx, source):
        raise NotImplementedError("Subclasses must implement `create_frame_generator`")

    def __iter__(self):
        return self

    def __next__(self):
        start_time = time.time()
        try:
            frames = [next(frame_gen) for frame_gen in self._frame_generators]
            if not frames or any(f is None for f in frames):
                raise StopIteration
            for frame, mat in zip(frames, self.output_mats):
                mat.unsafe_update_mat(frame)

            if self.fps > 0:
                elapsed = time.time() - start_time
                sleep_time = self._min_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            return self.output_mats
        except StopIteration:
            raise StopIteration

    def reset_generators(self):
        self.release_resources()
        self._frame_generators = [self.create_frame_generator(i, src) for i, src in enumerate(self.sources)]

    def release(self):
        for mat in self.output_mats:
            mat.release()
        self.release_resources()

    def __del__(self):
        self.release()

    def __len__(self):
        return None  # Streaming generator — length not defined
    
class NumpyRawFrameFileGenerator(PointCloudMatGenerator):
    shape_types: List['ShapeType']
    loop:bool=True
    def create_frame_generator(self, idx,source):
        arr = np.load(source)
        def gen(arr=arr):
            cnt=0
            while True:
                # idx = np.random.choice(len(arr))
                # yield np.ascontiguousarray(arr[20])
                if self.loop:
                    idx = cnt%len(arr)
                else:
                    idx = cnt
                    if idx>=len(arr):
                        break
                d = arr[idx]
                d = d[~np.isnan(d.sum(1))]
                yield np.ascontiguousarray(d)
                cnt+=1
        return gen()

RosPointCloud2Generator = None
try:
    import rospy
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs import point_cloud2 as pc2

    class _SubWrapper:
        """Small wrapper so your generic resource cleanup finds a .close()."""
        def __init__(self, sub):
            self._sub = sub
        def close(self):
            try:
                self._sub.unregister()
            except Exception:
                pass

    class RosPointCloud2Generator(PointCloudMatGenerator):
        """
        Read PointCloud2 from ROS Noetic topics.

        sources: List[str]  -> ROS topic names (e.g. "/lidar/points")
        shape_types: List[ShapeType] -> one per source (same as your base)
        """
        queue_size: int = 4
        ros_node_name: str = "RosPointCloud2Generator"

        def _ensure_ros(self):
            if rospy is None or pc2 is None:
                raise ImportError(
                    "ROS (rospy, sensor_msgs.point_cloud2) not available. "
                    f"Import error: {_ros_import_error!r}"
                )
            # init_node can only be called once per process; guard it
            if not rospy.core.is_initialized():
                # disable_signals=True so it plays nice inside libraries/subprocesses
                rospy.init_node(f"{self.ros_node_name}_{self.uuid.replace(':','_')}",
                                anonymous=True, disable_signals=True)

        def _msg_to_numpy(self, msg: PointCloud2) -> np.ndarray:
            """
            Convert PointCloud2 -> (N,3) float32 array [x,y,z], drop NaNs.
            """
            # Fast path: read x,y,z; skip NaNs at source
            gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            # np.fromiter is ~2x faster than np.array(list(...)) for large clouds
            arr = np.fromiter(gen, dtype=np.float32, count=-1)
            if arr.size == 0:
                return np.empty((0, 3), dtype=np.float32)
            arr = arr.reshape((-1, 3))
            # Ensure contiguous (your code calls np.ascontiguousarray downstream)
            return np.ascontiguousarray(arr)

        def create_frame_generator(self, idx, source) -> Iterator[np.ndarray]:
            """
            source is a ROS topic name publishing sensor_msgs/PointCloud2.
            """
            self._ensure_ros()

            q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.queue_size)

            def cb(msg: PointCloud2):
                try:
                    frame = self._msg_to_numpy(msg)
                    # Drop oldest if full to avoid stalling callback thread
                    try:
                        q.put_nowait(frame)
                    except queue.Full:
                        _ = q.get_nowait()
                        q.put_nowait(frame)
                except Exception as e:
                    logger(f"[{self.uuid}] PointCloud2 conversion error on {source}: {e}")

            sub = rospy.Subscriber(source, PointCloud2, cb, queue_size=1)
            # Ensure we clean it up later
            self.register_resource(_SubWrapper(sub))

            def gen(qref=q):
                # Keep yielding as long as ROS is alive; this is a live stream
                while not rospy.is_shutdown():
                    try:
                        frame = qref.get(timeout=1.0)
                        yield frame
                    except queue.Empty:
                        # No data yet; keep trying (streaming source)
                        continue

            return gen()

except Exception as e:
    rospy = None
    PointCloud2 = None
    pc2 = None
    _ros_import_error = e
    print(f"ROS PointCloud2 generator not available: {e}")

class PointCloudMatGenerators(BaseModel):

    @staticmethod
    def dumps(gen:PointCloudMatGenerator):
        return json.dumps(gen.model_dump())
    
    @staticmethod
    def loads(gen_json:str)->PointCloudMatGenerator:
        gen = {
            'NumpyRawFrameFileGenerator':NumpyRawFrameFileGenerator,
            'RosPointCloud2Generator': RosPointCloud2Generator,
        }
        g = json.loads(gen_json)
        return gen[f'{g["uuid"].split(":")[0]}'](**g) 

    @staticmethod
    def worker(gen_serialized):
        gen = PointCloudMatGenerators.loads(gen_serialized)
        for imgs in gen: pass
        
    @staticmethod
    def run_async(gen: 'PointCloudMatGenerator | str'):
        if isinstance(gen, str):
            gen_serialized = gen
        else:
            gen_serialized = gen.model_dump_json()

        p = multiprocessing.Process(target=PointCloudMatGenerators.worker, args=(gen_serialized,))
        p.start()
        return p