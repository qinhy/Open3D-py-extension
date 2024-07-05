from open3dpypro import PointCloud

pcd = PointCloud().read_pcd('./data/bunny.pcd')
print(pcd.size())
pcd = PointCloud().read_las('./data/bunny.las')
print(pcd.size())
pcd = PointCloud().read_e57('./data/bunny.e57')
print(pcd.size())