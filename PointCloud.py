import os
import threading
import open3d as o3d
import numpy as np
from typing import Callable, List

class PointCloudBase:
    COLOR_CHART = np.asarray([[230,0,18],[233,49,9],[235,97,0],[239,125,0],[243,152,0],[248,176,0],[252,200,0],[254,226,0],[255,251,0],[231,235,0],[207,219,0],[175,207,16],[143,195,31],[89,184,44],[34,172,56],[17,163,62],[0,153,68],[0,154,88],[0,155,107],[0,157,129],[0,158,150],[0,159,172],[0,160,193],[0,160,213],[0,160,233],[0,147,221],[0,134,209],[0,119,196],[0,104,183],[0,88,170],[0,71,157],[15,52,147],[29,32,136],[63,29,135],[96,25,134],[121,16,133],[146,7,131],[168,4,130],[190,0,129],[209,0,128],[228,0,127],[229,0,117],[229,0,106],[229,0,93],[229,0,79],[230,0,65],[230,0,51]])
    def __init__(self, xyz: np.ndarray = None, rgb: np.ndarray = None, normals: np.ndarray = None, intensity: np.ndarray = None, labels: np.ndarray = None, row_index: np.ndarray = None, column_index: np.ndarray = None, e57: np.ndarray = None):
        class ThreadLock:
            def __init__(self):
                self.lock = threading.Lock()
            def __enter__(self):
                self.lock.acquire()
            def __exit__(self, type, value, traceback):
                self.lock.release()
        self.pcd_lock = ThreadLock()

        self.e57 = None
        self.intensity = []
        self.labels = []
        self.row_index = []
        self.column_index = []
        self.pcd:o3d.geometry.PointCloud = None

        if type(xyz) is o3d.geometry.PointCloud:
            self.pcd = xyz
        else:
            self.pcd = o3d.geometry.PointCloud()
            if intensity is not None and len(intensity)>0 and np.prod(intensity.shape)>0:
                self.set_intensity(intensity)
                
            if row_index is not None and len(row_index)>0 and np.prod(row_index.shape)>0:
                self.row_index = row_index
            
            if column_index is not None and len(column_index)>0 and np.prod(column_index.shape)>0:
                self.column_index = column_index

            if xyz is not None and np.prod(xyz.shape)>0:
                self.set_points(xyz)
                
            if rgb is not None and len(rgb)>0 and np.prod(rgb.shape)>0:
                if rgb.dtype == np.uint8:
                    rgb = rgb/255.0
                self.set_rgb(rgb)
                
            if normals is not None and len(normals)>0:
                self.set_normals(normals)

        self.pcd_tree = None
        self.get_center = self.pcd.get_center
        self.has_colors = self.pcd.has_colors
        self.has_normals = self.pcd.has_normals
        self.has_points = self.pcd.has_points
        self.is_empty = self.pcd.is_empty
        self.rotate = self.pcd.rotate
        self.segment_plane = self.pcd.segment_plane
        self.scan_No = -1
    
    def __del__(self):
        if self.e57:self.e57.close()

    def clear(self):
        self.pcd = o3d.geometry.PointCloud()
        # self.init()

    def size(self):
        return len(self.pcd.points)
    
    def transform(self, T: np.ndarray):
        self.pcd.transform(T)
        return self

    def translate(self, t: np.ndarray):
        self.pcd.translate(t)
        return self           
        
    def has_rgb(self):
        return self.has_points() and self.size() == len(self.pcd.colors)

    def has_intensity(self):
        return self.has_points() and self.size() == len(self.intensity)

    def has_labels(self):
        return self.has_points() and self.size() == len(self.labels)
    
    def has_col_row(self):
        return self.has_points() and self.size() == len(self.column_index) == len(self.row_index)
    
    def set_rgb(self, colors: np.ndarray):
        assert colors.shape[1] == 3,'points shape must be (n,3)'
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        return self
    
    def set_points(self, points: np.ndarray):
        assert points.shape[1] == 3,'points shape must be (n,3)'
        self.pcd.points = o3d.utility.Vector3dVector(points)
        return self
    
    def set_normals(self, normals: np.ndarray):
        assert normals.shape[1] == 3,'normals shape must be (n,3)'
        self.pcd.normals = o3d.utility.Vector3dVector(normals)
        return self

    def set_intensity(self, intensity: np.ndarray):
        assert intensity.shape[1] == 1,'intensity shape must be (n,1)'
        self.intensity = intensity
        return self

    def set_labels(self, labels: np.ndarray):
        assert labels.shape[1] == 1,'labels shape must be (n,1)'
        self.labels = labels
        return self
    
    def get_points(self):
        return np.asarray(self.pcd.points)
    
    def get_colors(self):
        return np.asarray(self.pcd.colors)
    
    def get_normals(self):
        return np.asarray(self.pcd.normals)

    def get_intensity(self):
        return self.intensity
    
    def get_labels(self):
        return self.labels
    
    def get_col_row(self):
        return (self.column_index,self.row_index)

    def set_uniform_label(self, label: int):
        self.labels = np.ones((self.size(),1),dtype=int)*label
        return self
    
    def set_uniform_intensity(self, intensity: float):
        self.set_intensity(np.ones((self.size(),1),dtype=int)*intensity)
        return self

    def get_aabb(self):
        return self.pcd.get_min_bound(),self.pcd.get_max_bound()
    
    def calc_KDtree(self):
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        return self.pcd_tree
    
    def get_KDtree(self):
        if self.pcd_tree is None:
            self.calc_KDtree()
        return self.pcd_tree
    
    def get_points_by_knn(self, point_idx: int, max_nn: int = 1000000) -> np.ndarray:
        [k, idx, _] = self.get_KDtree().search_knn_vector_3d(self.pcd.points[point_idx], max_nn)
        return k, idx, _
    
    def get_points_radius(self, point_idx: int, search_radius: float = 30.0) -> np.ndarray:
        [k, idx, _] = self.get_KDtree().search_radius_vector_3d(self.pcd.points[point_idx], search_radius)
        return k, idx, _

    def read_pcd(self, filename: str, format: str = 'auto', remove_nan_points: bool = False, remove_infinite_points: bool = False, print_progress: bool = False):
        return self.__class__(o3d.io.read_point_cloud(filename=filename, format=format, remove_nan_points=remove_nan_points, remove_infinite_points=remove_infinite_points, print_progress=print_progress))
        
    def save_pcd(self, filename: str, write_ascii: bool = False, compressed: bool = False, print_progress: bool = False):
        return o3d.io.write_point_cloud(filename=filename, pointcloud=self.pcd, write_ascii=write_ascii,
                  compressed=compressed, print_progress=print_progress)
    
    def clone(self):
        return self.select_by_index(np.arange(self.size()))

    def select_by_index(self, indices: np.ndarray, invert: bool = False):
        res = self.__class__()
        if invert:
            idx = np.ones(len(self.pcd.points),dtype=bool)
            idx[indices] = False
        else:
            idx = np.zeros(len(self.pcd.points),dtype=bool)
            idx[indices] = True
            
        if self.has_points():
            res.set_points(self.get_points()[idx])
        if self.has_colors():
            res.set_rgb(self.get_colors()[idx])
        if self.has_normals():
            res.set_normals(self.get_normals()[idx])
        if self.has_intensity():
            res.intensity = self.intensity[idx].copy()
        if self.has_labels():
            res.labels = self.labels[idx].copy()
        return res
    
    def draw(self, geos: List[o3d.geometry.Geometry] = [], point_size: int = 1, centralize: bool = False) -> None:
        geos += [self]
        geos = [(i.pcd if hasattr(i, 'pcd') else i) for i in geos]
        if centralize:
            cent = np.asarray([p.get_center() for p in geos]).mean(0)
            for p in geos:
                p.translate(cent)
        o3d.visualization.draw([o3d.geometry.TriangleMesh.create_coordinate_frame()]+geos,point_size=point_size)
        return self
    
class PointCloudSelections(PointCloudBase):
    def select_by_box(self, center, x_direction, y_direction, z_direction, invert=False):
        # Normalize direction vectors and calculate squared norms (for half-lengths of the box sides)
        x_direction = np.asarray(x_direction)
        y_direction = np.asarray(y_direction)
        z_direction = np.asarray(z_direction)
        center = np.asarray(center)
        
        x_direction /= np.linalg.norm(x_direction)
        y_direction /= np.linalg.norm(y_direction)
        z_direction /= np.linalg.norm(z_direction)

        x_half_length = np.linalg.norm(x_direction) ** 2
        y_half_length = np.linalg.norm(y_direction) ** 2
        z_half_length = np.linalg.norm(z_direction) ** 2
        
        # Convert Open3D point cloud to numpy array
        points = self.get_points()
        
        # Calculate the direction vectors from the center to each point
        dir_vectors = points - center

        # Check each point's position relative to the box defined by the direction vectors and half-lengths
        in_box = np.logical_and.reduce((
            np.square(dir_vectors @ x_direction) < x_half_length,
            np.square(dir_vectors @ y_direction) < y_half_length,
            np.square(dir_vectors @ z_direction) < z_half_length
        ))
        
        # Create a new point cloud with the selected points
        selected_pcd = self.select_by_bool(in_box,invert)

        return selected_pcd
    
    def select_by_normals_index(self, comfunc=lambda ns: np.ones(len(ns),dtype=bool)):
        if not self.pcd.has_normals():
            self.pcd.estimate_normals()
        res = comfunc(np.asarray(self.pcd.normals))
        return np.where(res)[0]
    
    def select_by_normals(self, comfunc: Callable[[np.ndarray], np.ndarray] = lambda ns: np.ones(len(ns), dtype=bool)) -> None:
        return self.select_by_index(self.select_by_normals_index(comfunc=comfunc))

    def select_by_normals_cosine_index(self, model: np.ndarray, similarity: float = 0.99) -> None:
        comp = lambda ns:(ns@model[:3])/(np.linalg.norm(ns,axis=1) *np.linalg.norm(model[:3]))>=similarity
        return self.select_by_normals_index(comfunc=comp)
    
    def select_by_normals_cosine(self, model: np.ndarray, similarity: float = 0.99) -> None:
        return self.select_by_index(self.select_by_normals_cosine_index(model=model,similarity=similarity))
    
    def select_by_colors_index(self, comfunc: Callable[[np.ndarray], np.ndarray] = lambda ns: np.ones(len(ns), dtype=bool)) -> None:
        res = comfunc(np.asarray(self.pcd.colors))
        return np.where(res)[0]
    
    def select_by_colors_cosine_index(self, model: np.ndarray, similarity: float = 0.99) -> None:
        comp = lambda ns:(ns@model[:3])/(np.linalg.norm(ns,axis=1) *np.linalg.norm(model[:3]))>=similarity
        return self.select_by_colors_index(comfunc=comp)
    
    def select_by_colors_cosine(self, model: np.ndarray, similarity: float = 0.99) -> None:
        return self.select_by_index(self.select_by_colors_cosine_index(model=model,similarity=similarity))

    def select_by_sphere(self, center: np.ndarray, radius: float = 0.1) -> None:
        return self.select_by_index(self.select_by_sphere_index(center,radius))

    def select_by_sphere_index(self, center: np.ndarray, radius: float = 0.1) -> None:
        return self.select_by_XYZ_index(lambda Xs,Ys,Zs:((Xs-center[0])**2 + (Ys-center[1])**2 + (Zs-center[2])**2)**0.5 < radius)

    def select_by_XYZ_index(self, comfunc: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = lambda Xs, Ys, Zs: np.ones(len(Xs), dtype=bool)) -> None:
        ps = np.asarray(self.pcd.points)
        Xs,Ys,Zs = ps[:,0],ps[:,1],ps[:,2]
        res = comfunc(Xs,Ys,Zs)
        return np.where(res)[0]
    
    def select_by_XYZ(self, comfunc: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = lambda Xs, Ys, Zs: np.ones(len(Xs), dtype=bool)) -> None:
        return self.select_by_index(self.select_by_XYZ_index(comfunc))
    
    def select_by_plane(self, model: np.ndarray, thickness: float = 0.03, invert: bool = False) -> None:
        p = np.asarray(self.pcd.points)
        a,b,c,d = model
        abc = np.asarray([a,b,c]).reshape(3,1)
        if type(thickness) is tuple:
            res = (p @ abc + d)/(a**2+b**2+c**2)**0.5
            res =  np.logical_and(res > thickness[0], res < thickness[1])
            if invert:
                res = np.logical_not(res)
        else :
            res = (np.abs(p @ abc + d)/(a**2+b**2+c**2)**0.5) < thickness
            if invert:
                res = np.logical_not(res)
        idx = np.where(res)[0]
        return self.select_by_index(idx),idx
    
    def select_by_aabb_list_index(self, aabb_min_max_list: np.ndarray):
        points = self.get_points()
        is_inside = np.ones_like(points,dtype=bool)
        for aabb_min, aabb_max in aabb_min_max_list:
            aabb_min, aabb_max = np.asarray(aabb_min), np.asarray(aabb_max)
            is_inside = np.logical_or(np.logical_and((points >= aabb_min) , (points <= aabb_max)), is_inside)
        is_inside = np.all(is_inside, axis=1)
        return np.arange(self.size())[is_inside]
    
    def select_by_aabb_list(self, aabb_min: np.ndarray, aabb_max: np.ndarray):
        return self.select_by_index(self.select_by_aabb_list_index(aabb_min, aabb_max))

    def select_by_aabb_index(self, aabb_min: np.ndarray, aabb_max: np.ndarray):
        points = self.get_points()
        aabb_min, aabb_max = np.asarray(aabb_min), np.asarray(aabb_max)
        is_inside = np.all(np.logical_and((points >= aabb_min) , (points <= aabb_max)), axis=1)
        return np.arange(self.size())[is_inside]
    
    def select_by_aabb(self, aabb_min: np.ndarray, aabb_max: np.ndarray):
        return self.select_by_index(self.select_by_aabb_index(aabb_min, aabb_max))

    def select_by_topN(self, n: int):
        return self.select_by_index(np.arange(self.size())[:n])
    
    def select_by_bool(self, indices: np.ndarray, invert: bool = False):
        tmp = indices
        if invert:
            tmp = np.logical_not(indices)
        return self.select_by_index(np.arange(self.size())[tmp])

    def select_by_radius_index(self, r:float):
        ps = np.asarray(self.pcd.points)
        Xs,Ys,Zs = ps[:,0],ps[:,1],ps[:,2]
        rs = (Xs**2+Ys**2+Zs**2)**0.5
        return np.where(rs<=r)[0]
    
    def select_by_radius(self, r:float):
        return self.select_by_index(self.select_by_radius_index(r))

class PointCloudUtility(PointCloudSelections):
    def create_from_sphere(self,r=1.0,cent=[0,0,0],number_of_points=10000):
        pcd = o3d.geometry.TriangleMesh.create_sphere(r
                ).sample_points_uniformly(number_of_points=number_of_points, use_triangle_normal=False)
        return self.__class__(pcd).translate(cent)
    
    def paint_uniform_color(self, color: list = [1., 0., 0.]):
        self.pcd.paint_uniform_color(color)
        return self
    
    def split_by_labels(self):
        ul = np.unique(self.labels)
        pcds = []
        for j in ul:
            pcds.append(self.select_by_bool(self.labels==j))
        return pcds,ul
                  
    def estimate_normals(self, param=o3d.geometry.KDTreeSearchParamKNN(30)):
        self.pcd.estimate_normals(param)
        return self
      
    def centralize(self):
        self.translate(-self.get_center())
        return self
        
    def voxel_down_sample_and_trace(self, voxel_size: float):
        pcd,idxmat,vec = self.pcd.voxel_down_sample_and_trace(voxel_size=voxel_size,
                        min_bound=self.pcd.get_min_bound(),max_bound=self.pcd.get_max_bound(), approximate_class=False)
        return self.__class__(pcd),idxmat,vec
    
    def random_down_sample(self, down_sample_ratio: float = 0.1):
        if down_sample_ratio<=0 or down_sample_ratio > 1:
            return self.__class__()
        random_down_sample_pcd = self.pcd.random_down_sample(down_sample_ratio)
        return self.__class__(random_down_sample_pcd)
    
    def uniform_down_sample(self, down_sample_ratio: float = 0.1):
        if down_sample_ratio<=0 or down_sample_ratio > 1:
            return self.__class__()
        uniform_down_sample_pcd = self.pcd.uniform_down_sample(int(1/down_sample_ratio))
        return self.__class__(uniform_down_sample_pcd)
        
    def create_voxel(self, voxel_size: float = 0.05):
        voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd,voxel_size=voxel_size)
        return voxel
    
    def voxel_down_sample(self, voxel_size: float):
        return self.__class__(self.pcd.voxel_down_sample(voxel_size))

    def read_mesh_as_pcd(self, path: str, sample_points_uniformly=None):
        if sample_points_uniformly is None:
            return self.__class__(np.asarray(o3d.io.read_triangle_mesh(path).vertices))
        else:
            return self.__class__(o3d.io.read_triangle_mesh(path).sample_points_uniformly(number_of_points=sample_points_uniformly))

    def remove_statistical_outlier(self, nb_neighbors: int = 20, std_ratio: float = 2.0, print_progress: bool = False):
        res = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio, print_progress=print_progress)
        return self.select_by_index(res[1]),res[1]
    
    def merge_pcds(self, raw_pcds: np.ndarray, rgb: bool = False, intensity: bool = False, normals: bool = False, labels: bool = False):
        pcds = [p if hasattr(p,'pcd') else self.__class__(p) for p in raw_pcds]        
        ps = np.vstack([ np.asarray(pcd.pcd.points) for pcd in pcds])
        pcdres = self.__class__(ps)
        
        if rgb:        
            ps = np.vstack([ pcd.get_colors() if pcd.has_colors() else np.ones((pcd.size(),3),dtype=float) for pcd in pcds])
            pcdres.set_rgb(ps)
        if normals:        
            for pcd in pcds:
                if pcd.has_normals():continue
                pcd.estimate_normals()
            ps = np.vstack([ pcd.get_normals() for pcd in pcds])
            pcdres.set_normals(ps)

        if intensity:            
            pcdres.set_intensity(np.vstack([ pcd.get_intensity() if pcd.has_intensity() else pcd.set_uniform_intensity(0).get_intensity() for pcd in pcds]))
            
        if labels:          
            pcdres.labels = np.vstack([ pcd.get_labels() if pcd.has_labels() else pcd.set_uniform_label(0).get_labels()  for pcd in pcds])

        return pcdres
    
    def append_pcd(self, pcd: o3d.geometry.PointCloud):
        self.pcd = self.merge_pcds([self,pcd])
        # self.init()
        return self
    
    def distance2plane(self, plane: np.ndarray) -> np.ndarray:
        pcdp = np.asarray(self.pcd.points)
        a,b,c,d = plane
        d = (pcdp*np.asarray([a,b,c])).sum(1)+d
        d = d/(a**2+b**2+c**2)**0.5
        return d
        
    def remove_plane_outlier(self, plane_model: np.ndarray, thickness: float = 0.03, similarity: float = 0.999, invert: bool = False) -> None:
        pch2idx = self.select_by_normals_index(lambda ns:((ns@plane_model[:3])/(np.linalg.norm(ns,axis=1) *np.linalg.norm(plane_model[:3])))>=similarity)#maskByNormls(pc,nor=plane_model[:3],threshold=similarity) & res 
        tmp,planeidx = self.select_by_index(pch2idx).select_by_plane(plane_model,thickness)
        flooridx = np.arange(len(self.pcd.points))[pch2idx][planeidx]     
        floor = self.select_by_index(flooridx)
        return floor,flooridx

    def project2plane(self, plane: np.ndarray) -> None:
        plane = np.asarray(plane)
        plane[:3] = plane[:3]#/np.linalg.norm(plane[:3])
        ps = np.asarray(self.pcd.points)
        dis = self.distance2plane(plane)
        dis = dis.reshape(-1,1)*plane[:3].reshape(1,3)
        return self.__class__(ps-dis)
    
    def seg_plane_by_svd(self):
        centroid = self.get_points().mean(axis=0)
        # Center the points around the centroid
        points_centered = self.get_points() - centroid
        # Perform singular value decomposition
        u, s, v = np.linalg.svd(points_centered)
        # The normal vector of the plane is the last column of v
        a,b,c = normal = v[-1]
        d = -sum(centroid * normal)
        # print(f"The equation of the plane is {a}x + {b}y + {c}z + {d} = 0")
        return a, b, c, d
    
class PointCloud(PointCloudUtility):
    try:
        from PIL import Image
        def _open_img(self, path: str):
            from PIL import Image
            img = Image.open(path)
            assert self.size()==np.prod(img.size)
            img = np.asarray(img)
            return img

        def load_rgb_from_img(self, path: str, upsidedown: bool = False):
            img = self._open_img(path)
            if upsidedown: img = img[::-1]
            self.set_rgb(img.reshape(-1,3)/255.0)
        
        def load_label_from_img(self, path: str, upsidedown: bool = False):
            img = self._open_img(path)
            if upsidedown: img = img[::-1]
            self.set_labels(img.reshape(-1,1))
        
        def load_intensity_from_tiff(self, path: str, upsidedown: bool = False):
            img = self._open_img(path)
            if upsidedown: img = img[::-1]
            self.set_intensity(img.reshape(-1,1))
        
        
        def _save_img(self, name: str, src: np.ndarray, c: np.ndarray, dtype=np.uint8):
            from PIL import Image
            if dtype==np.uint8:           
                src = (src-src.min())/(src.max()-src.min())
                src = (src.reshape(-1,c)*255).astype(dtype)

            img_shape = (self.row_index.max()+1,self.column_index.max()+1,c)
            img = np.zeros(img_shape,dtype=dtype).reshape(-1,c)
            img[self.row_index.astype(int) * img_shape[1] + self.column_index.astype(int)] = src
            img = img.reshape(*img_shape) if c>1 else img.reshape(*img_shape[:2])
            img = Image.fromarray(img)
            img.save(name)

        def save_image(self, filepath: str = '.', rgb: bool = False, intensity: bool = False, normals: bool = False, depth: bool = False, depth_centralize: bool = False):
            ext = filepath.split('.')[-1]
            if ext=='jpg':
                dtype=np.uint8
            if ext=='png':
                dtype=np.uint8
            if ext=='tiff':
                dtype=np.float32

            assert self.has_col_row()
            if rgb:self._save_img(filepath+'/rgb.'+ext ,self.get_colors(),3,dtype=np.uint8)
            if intensity:self._save_img(filepath+'/intensity.'+ext ,self.get_intensity(),1,dtype=dtype)
            if normals:
                if not self.has_normals():self.estimate_normals()
                self._save_img(filepath+'/normals.'+ext ,self.get_normals(),3,dtype=dtype)
            if depth:
                depthmat = np.linalg.norm(self.get_points(),axis=1)
                if depth_centralize:
                    depthmat -= depthmat.mean(0)
                self._save_img(filepath+'/depth.'+ext ,depthmat,1,dtype=dtype)
                    
    except Exception as e:
        print(e,'no PIL, can not do save as img func of [save_png , save_jpg , save_tiff]')

    
    try :
        import laspy
        def get_lasdata(self, h: laspy.LasHeader, pr: laspy.PackedPointRecord = None, pt_src_id: bool = False):
            import laspy
            las = laspy.LasData(h,pr)
            ps = self.get_points()
            las.x = ps[:, 0].astype(np.float32)
            las.y = ps[:, 1].astype(np.float32)
            las.z = ps[:, 2].astype(np.float32)
            if self.has_rgb():
                rgb = np.asarray(self.pcd.colors)
                las.red   = (rgb[:, 0]*255).astype(np.uint8)
                las.green = (rgb[:, 1]*255).astype(np.uint8)
                las.blue  = (rgb[:, 2]*255).astype(np.uint8)
            if self.has_intensity():
                las.intensity = (self.get_intensity().flatten()*255).astype(np.uint8)
            if pt_src_id:
                assert hasattr(self, 'e57') and self.e57.all_points == len(ps)
                idx = [0] + [sum(self.e57.point_counts[0:i]) for i in range(len(self.e57.point_counts))]
                las.pt_src_id = np.zeros(len(ps),dtype=np.uint16)
                for i,ii in enumerate(zip(idx[:-1],idx[1:])):
                    las.pt_src_id[ii[0]:ii[1]] = i
            if self.has_labels():
                las.raw_classification = self.labels.flatten().astype(np.uint8)
            return las

        def read_las(self, path: str, rgb: bool = False, intensity: bool = False, labels: bool = False):
            import laspy
            with laspy.open(path) as fh:
                las = fh.read()
                print(las)
                lp = las.points
                xyz = np.asarray([lp.x,lp.y,lp.z]).T
                rgb = np.asarray([lp.red,lp.green,lp.blue]).T/(2.0**16) if rgb else []
                intensity = np.asarray(lp.intensity).reshape(-1,1) if intensity else []
                labels = np.asarray(lp.raw_classification).reshape(-1,1) if labels else []
            return self.__class__(xyz=xyz,rgb=rgb,intensity=intensity,labels=labels)
        
        def read_las_gen(self, path: str, every_k_points: int = 100_0000, rgb: bool = False, intensity: bool = False, labels: bool = False):
            import laspy
            with laspy.open(path) as file:
                count = 0
                for points in file.chunk_iterator(every_k_points):
                    xyz = np.vstack([points.x,points.y,points.z]).T
                    rgb = np.asarray([points.red,points.green,points.blue]).T/(2.0**16) if rgb else []
                    intensity = np.asarray(points.intensity).reshape(-1,1) if intensity else []
                    labels = np.asarray(points.raw_classification).reshape(-1,1) if labels else []         
                    pcd = self.__class__(xyz=xyz,rgb=rgb,intensity=intensity,labels=labels)
                    count += pcd.size()
                    print(f"[read_las_gen]: read {count / file.header.point_count * 100}% points.")
                    yield pcd
        
        def append_save_las(self, path: str = './pcd.las', pt_src_id: bool = False):
            import laspy
            assert os.path.isfile(path),f'file {path} is no exist.'
            with laspy.open(path) as fh:
                las = fh.read()
                pf = las.point_format
            packed_point_record = laspy.PackedPointRecord.zeros(self.size(),pf)
            lasdata = self.get_lasdata(las.header,packed_point_record)
            with laspy.open(path, mode='a') as outlas:
                outlas.append_points(lasdata.points)

        def save_las(self, path: str = './pcd.las', pt_src_id: bool = False):
            import laspy
            h = laspy.LasHeader(point_format=3, version="1.2")
            h.scales = np.array([0.0001,0.0001,0.0001])
            las = self.get_lasdata(h)
            las.write(path)
    except Exception as e:
        print(e,'can not do laspy IO!')
    
    
    def split_pcd_index(self, nn: int, random: bool = False):
        idx = self.size()
        if idx<=nn:return [np.arange(idx)]
        idxs = np.ones(idx//nn)*nn+np.asarray(list(map(np.sum,np.array_split([1]*(idx%nn),(idx//nn)))))
        idxs = idxs.astype(int)
        samples= []
        r = np.arange(idx)
        if random : np.random.shuffle(r)
        tmp = [idxs[:i].sum() for i in range(len(idxs))]
        samples = [ r[start:start+num] for start,num in zip(tmp,idxs)]
        return samples

    def split_pcd(self, nn: int, random: bool = False):
        if self.size()<=nn:return [self]
        samples = self.split_pcd_index(nn=nn,random=random)
        pcds = [self.select_by_index(i) for i in samples]
        return pcds
    
    def split_by_voxel(self, voxel_size: float = 0.01, random: bool = True, top_n: int = 10):
        pcds = []
        pcdv = self.voxel_down_sample_and_trace(voxel_size)[2]
        pcdv = [np.asarray(v).tolist() for v in pcdv]
        if random:
            for v in pcdv:
                if len(v)>0:
                    np.random.shuffle(v)
        pcd_idxs = []
        while True:
            cnt=0
            pcd_idx = []
            for i,v in enumerate(pcdv):
                if len(v)>0:
                    cnt+=1
                    pcd_idx.append(v.pop())
            if cnt==0:break
            pcd_idxs += pcd_idx
            pcds.append(self.select_by_index(pcd_idx))
            if len(pcds)>=top_n:
                break
        other = self.select_by_index(pcd_idxs,True)
        return pcds,other
    
    def rotation_matrix_from_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b= (vec1 /np.linalg.norm(vec1)).reshape(3), (vec2 /np.linalg.norm(vec2)).reshape(3)
        v= np.cross(a, b)
        if any(v): #if not all zeros then
            c= np.dot(a, b)
            s= np.linalg.norm(v)
            kmat= np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 -c) /(s ** 2))
        else:
            return np.eye(3) #cross of all zeros only occurs on identical directions

    def rotate_by_normal(self, plane: np.ndarray) -> None:
        a,b,c,d = plane
        T = np.eye(4)
        T[2,3] = d
        z_axis = np.array([a,b,c])/np.linalg.norm(np.array([a,b,c]))
        R = self.rotation_matrix_from_vectors(z_axis, (0,0,1))
        T[:3,:3] = R
        self.pcd.transform(T)
        return self,T

    def to_2D_Img(self, resolution: float = 0.05, color: bool = False) -> np.ndarray:
        usecolor = False
        if color and len(self.pcd.colors)>0:
            usecolor = True
        if color and not usecolor:
            print('has no rgb data, cannot use color!')
        points2d = np.asarray(self.pcd.points)
        minx,miny = points2d[:,0].min(),points2d[:,1].min()
        T =   np.asarray([[1., 0., 0., -minx],
                           [0., 1., 0., -miny],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
        
        T =   np.asarray([[1/resolution, 0., 0., 0.],
                           [0., 1/resolution, 0., 0.],
                           [0., 0., 1/resolution, 0.],
                           [0., 0., 0., 1.]]) @ T    
        
        R,t = T[:3,:3],T[:3,3]
        tmp = ( (R @ points2d.T).T + t).round().astype(int)
        ix,iy = tmp[:,0],tmp[:,1]
        sizex,sizey  = ix.max(),iy.max()
        
        ix[ix<0] = 0
        ix[ix>(sizex-1)] = sizex-1
        iy[iy<0] = 0
        iy[iy>(sizey-1)] = sizey-1
        if usecolor:
            rgb = np.asarray(self.pcd.colors)
            img = np.zeros((sizey,sizex,3),np.uint8)
            img[iy,ix] = (rgb*255).astype(np.uint8)
        else:
            img = np.zeros((sizey,sizex),np.uint8)
            img[iy,ix] = 255
        # if vis:
        #     plt.imshow(img)
        idx2img = list(zip(ix,iy))
        invT = np.linalg.inv(T)
        return img,minx,T,miny,invT,resolution,idx2img
        
    ############################# cv2 part ######################################
    try:
        import cv2
        def read_single_RGB(self, file: str, sampling_step: int = 10, resolution: float = 0.01, y_as_z: bool = True) -> None:
            import cv2
            img = cv2.imread(file)
            height,width = img.shape[:2]
            resolution_mat = np.eye(4)*resolution
            resolution_mat[-1,-1] = 1
            img2real = np.asarray( [[ 1. ,  0. ,  0. ,  0],
                                    [ 0. ,  -1.,  0. ,  height*resolution],
                                    [ 0. ,  0. ,  1. ,  0.    ],
                                    [ 0. ,  0. ,  0. ,  1.    ]])@resolution_mat
            if y_as_z:
                img2real = np.asarray([ [ 1. ,  0. ,  0. ,  0],
                                        [ 0. ,  0. ,  1. ,  0],
                                        [ 0. ,  1. ,  0. ,  0],
                                        [ 0. ,  0. ,  0. ,  1]])@img2real
            img = img[:,:,::-1].astype(float)/255.0
            pointsx,pointsy,pointsz = np.meshgrid(np.arange(0,width,sampling_step),
                                                  np.arange(0,height,sampling_step),[0])
            # print(pointsx.flatten().shape,pointsy.flatten().shape,pointsz.flatten().shape)
            colors = img.reshape(-1,3)[pointsy.flatten()*width+pointsx.flatten()].copy()
            points = np.vstack([pointsx.flatten(),pointsy.flatten(),pointsz.flatten()]).T
            pcd = self.__class__(points,rgb=colors)
            pcd.transform(img2real)
            return pcd
        
        def detect_3d_circles(self, plane: np.ndarray = np.asarray([0.95, -0.32, 0.02, 1.54]), thickness: float = 0.05, resolution: float = 0.005, minRadius: float = 0.045, maxRadius: float = 0.495, minInertiaRatio: float = 0.75, vis: bool = False) -> None:
            import cv2
            hsteelplane = plane    
            TT = np.eye(4)
            pcd,idx = self.select_by_plane(hsteelplane,thickness)
            seg3,T = pcd.rotate_by_normal(hsteelplane)
            TT = T @ TT
            res,minx,T,miny,invT,resolution,idx2img = seg3.to_2D_Img(resolution)
            TT = T @ TT
            res = cv2.morphologyEx(res,cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
            ret2, image = cv2.threshold(res, 0, 255, cv2.THRESH_OTSU)
            params = cv2.SimpleBlobDetector_Params()
            params.filterByInertia = True
            params.minInertiaRatio = minInertiaRatio# 0.3~0.9
            params.filterByArea = True
            params.minArea = int(np.pi*(0.1/resolution)**2)+1
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(image)
            blank = np.zeros((1, 1))
            blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if vis:
                cv2.imshow("Blobs Using Area", blobs)
                # cv2.imshow("Blobs Using Area", cv2.resize(blobs,(512,512)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if len(keypoints)==0:
                return []
            pts = [[i.pt[0], i.pt[1], 0, 1] for i in keypoints]
            pts = np.asarray(pts)
            pts = pts @ np.linalg.inv(TT).T
            r = np.asarray([i.size/2 * resolution for i in keypoints])
            pts[:,-1] = r  
            return pts
        
        def simple_seg_connected_components_by_img(self, img: np.ndarray, minx: float, T: float, miny: float, invT: float, resolution: float, idx2img: Callable[[np.ndarray], np.ndarray]) -> None:
            import cv2
            idx = np.asarray(idx2img)
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
            # print(nlabels)
            pcds = []
            p_labels = labels[idx[:,1],idx[:,0]]
            for i in stats[:,-1].argsort()[::-1][1:]:
                res = self.select_by_index(np.where(p_labels==i)[0].tolist())
                pcds.append(res)
            return pcds

        def simple_seg_connected_components(self, plane: np.ndarray, thickness: float = 0.05, resolution: float = 0.1, minpoints: int = 20, top_n: int = 100) -> None:
            import cv2
            floorMappcd,idx = self.select_by_plane(plane,thickness)
            if len(floorMappcd.pcd.points)==0:return []
            img,minx,T,miny,invT,resolution,idx2img = floorMappcd.to_2D_Img(resolution=resolution)
            idx = np.asarray(idx2img)
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
            # print(nlabels)
            pcds = []
            p_labels = labels[idx[:,1],idx[:,0]]
            for i in stats[:,-1].argsort()[::-1][1:top_n+1]:
                res = floorMappcd.select_by_index(np.where(p_labels==i)[0].tolist())
                if len(res.pcd.points) < minpoints:
                    continue
                pcds.append(res)
                # o3d.visualization.draw(pcds)
            return pcds
    except Exception as e:
        print(e,'can not do cv2 func of [detect_3d_circles , simple_seg_connected_components_by_img, simple_seg_connected_components, read_single_RGB')

    ############################# sklearn part ######################################
    try :
        from sklearn.cluster import DBSCAN as skDBSCAN
        def DBSCAN(self, eps: float = 0.05, min_samples: int = 3) -> None:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np.asarray(self.pcd.points))
            labels = clustering.labels_
            return [self.select_by_index(np.arange(len(labels))[labels==i])for i in range(len(set(labels)-{-1}))],labels
    except Exception as e:
        print(e,'can not do sklearn.cluster DBSCAN')
    
    def rotate_to_plane(self, plane: np.ndarray) -> None:
        a,b,c,d = plane
        T = np.eye(4)
        T[2,3] = d
        z_axis = np.array([a,b,c])/np.linalg.norm(np.array([a,b,c]))
        R = self.rotation_matrix_from_vectors(z_axis, (0,0,1))
        T[:3,:3] = R
        self.pcd.transform(T)
        return self,T
    
    def seg_planes(self, thickness: float = 0.01, ransac_n: int = 3, num_iterations: int = 450, top_n: float = 10e9, minPointsRatio: float = 0.1) -> None:
        """
        This method segments planes from the point cloud using RANSAC.

        Parameters:
        thickness (float): Distance threshold to count inliers. Default is 0.01.
        minPointsRatio (float): The minimum ratio of remaining points to original points to continue the loop. Default is 0.1.
        top_n (float): The maximum number of planes to segment. Default is 1e10.
        ransac_n (int): Number of points to sample in each RANSAC iteration. Default is 3.
        num_iterations (int): Number of RANSAC iterations. Default is 450.

        Returns:
        planes (list): A list of planes represented by the coefficients [a, b, c, d] of the plane equation ax+by+cz+d=0.
        pcds (list): A list of point clouds representing the points that were segmented into each plane.
        """
        # ... the rest of the function code ...

        cnt=-1    
        outlier_cloud,planes,pcds,aabbs = self,[],[],[]
        raw_size = self.size()
        if raw_size<ransac_n:return planes,pcds,aabbs        
        while outlier_cloud.size()/raw_size>minPointsRatio:
            cnt+=1             
            try:
                [a, b, c, d],inlines = outlier_cloud.segment_plane(distance_threshold=thickness,ransac_n=ransac_n,num_iterations=num_iterations)
            except Exception as e:
                print(e)
                break
            planes.append([a, b, c, d])
            inliners = outlier_cloud.select_by_index(inlines)
            
            #######################
            # cent = self.__class__(inliners.get_center().reshape(-1,3))
            # c1 = cent.project2plane(planes[-1])
            # c2 = cent.project2plane(inliners.random_down_sample(10000/inliners.size()).seg_plane_by_svd())
            # print('svd gap :',np.linalg.norm(c1.get_points()-c2.get_points()))
            #######################            
            
            pcds.append(inliners)
            aabbs.append(pcds[-1].get_aabb())
            outlier_cloud = outlier_cloud.select_by_index(inlines,True)
            if len(planes)>top_n:break#return planes,pcds,aabbs         
        pcds.append(outlier_cloud)
        aabbs.append(pcds[-1].get_aabb())
        return planes,pcds,aabbs




