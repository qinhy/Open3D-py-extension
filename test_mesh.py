import time
import open3d as o3d
import numpy as np

def read_point_cloud(file_path):
    print(f"[INFO] Reading point cloud from '{file_path}'...")
    start = time.time()
    pcd = o3d.io.read_point_cloud(file_path)
    count = len(pcd.points)
    duration = time.time() - start
    print(f"[INFO] Loaded {count} points in {duration:.3f} sec.")
    return pcd

def estimate_normals(pcd, method="default"):
    print(f"[INFO] Estimating normals ({method})...")
    start = time.time()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    if method == "poisson":
        pcd.orient_normals_consistent_tangent_plane(k=30)
    print(f"[INFO] Normals estimated in {time.time() - start:.3f} sec.")

def reconstruct_mesh_alpha_shape(pcd, alpha):
    print(f"[INFO] Reconstructing mesh using Alpha Shape (alpha = {alpha:.3f})...")
    start = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    duration = time.time() - start
    print(f"[INFO] Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles ({duration:.3f} sec).")
    return mesh

def reconstruct_mesh_bpa(pcd, radius):
    print(f"[INFO] Reconstructing mesh using Ball Pivoting (radius = {radius:.4f})...")
    start = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )
    duration = time.time() - start
    print(f"[INFO] Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles ({duration:.3f} sec).")
    return mesh

def reconstruct_mesh_poisson(pcd, depth=9, density_threshold=0.01):
    print(f"[INFO] Reconstructing mesh using Poisson (depth={depth})...")
    start = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    duration = time.time() - start
    print(f"[INFO] Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles ({duration:.3f} sec).")
    
    print("[INFO] Removing low-density vertices...")
    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, density_threshold)
    mesh = mesh.select_by_index(np.where(keep)[0])
    print(f"[INFO] Filtered mesh has {len(mesh.vertices)} vertices.")
    return mesh

def compute_normals(mesh):
    print("[INFO] Computing vertex normals...")
    start = time.time()
    mesh.compute_vertex_normals()
    print(f"[INFO] Normals computed in {time.time() - start:.3f} sec.")

def save_mesh(mesh, output_path):
    print(f"[INFO] Saving mesh to '{output_path}'...")
    start = time.time()
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"[INFO] Mesh saved in {time.time() - start:.3f} sec.")

def visualize_mesh(mesh):
    print("[INFO] Launching visualization window...")
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    print("[INFO] Visualization closed.")

def main(method="alpha"):
    total_start = time.time()

    input_pcd_path = './data/bunny.pcd'
    output_mesh_path = f'./data/output_mesh_{method}.ply'

    pcd = read_point_cloud(input_pcd_path)

    if method == "alpha":
        alpha = 0.03
        mesh = reconstruct_mesh_alpha_shape(pcd, alpha)

    elif method == "bpa":
        radius = 0.01
        estimate_normals(pcd)
        mesh = reconstruct_mesh_bpa(pcd, radius)

    elif method == "poisson":
        estimate_normals(pcd, method="poisson")
        mesh = reconstruct_mesh_poisson(pcd)

    else:
        raise ValueError(f"Unknown method: {method}")

    compute_normals(mesh)
    save_mesh(mesh, output_mesh_path)
    visualize_mesh(mesh)

    print(f"[INFO] Total execution time: {time.time() - total_start:.3f} sec.")

if __name__ == "__main__":
    # Options: "alpha", "bpa", "poisson"
    main(method="alpha")
    main(method="bpa")
    main(method="poisson")
