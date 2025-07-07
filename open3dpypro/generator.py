import json
import multiprocessing
import os
import sys
import glob
import time
import uuid
import platform
from enum import IntEnum
from typing import Iterator, List, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

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
            self.output_mats = [
                PointCloudMat(shape_type=shape_type).build(next(gen))
                for gen, shape_type in zip(self._frame_generators, self.shape_types)
            ]

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
                        print(f"Error during {method} on {res}: {e}")
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
    shape_types: list['ShapeType']
    def create_frame_generator(self, idx,source):
        arr = np.load(source)
        def gen(arr=arr):
            while True:
                idx = np.random.choice(len(arr))
                yield np.ascontiguousarray(arr[idx])
        return gen()

class PointCloudMatGenerators(BaseModel):

    @staticmethod
    def dumps(gen:PointCloudMatGenerator):
        return json.dumps(gen.model_dump())
    
    @staticmethod
    def loads(gen_json:str)->PointCloudMatGenerator:
        gen = {
            'NumpyRawFrameFileGenerator':NumpyRawFrameFileGenerator,
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