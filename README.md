# Open3D Python Extension

This project is a Python extension for the [Open3D](http.open3d.org/) library. It provides a `PointCloud` class that simplifies working with point cloud data, including reading from various file formats and performing common operations.

## Features

- Read point cloud data from `.pcd`, `.las`, and `.e57` files.
- Easy access to point coordinates, colors, normals, and intensity.
- Methods for common point cloud operations like transformations, normal estimation, and plane segmentation.
- A simple example script to demonstrate usage.

## Installation

To use this extension, you need to have Python and Open3D installed. You can install Open3D using pip:

```bash
pip install open3d
```

Then, you can clone this repository:

```bash
git clone https://github.com/your-username/Open3D-py-extension.git
cd Open3D-py-extension
```

## Usage

The `exmaple.py` script shows how to use the `PointCloud` class to read data from different file formats:

```python
from open3dpypro import PointCloud

# Read a .pcd file
pcd = PointCloud().read_pcd('./data/bunny.pcd')
print(f"Points in bunny.pcd: {pcd.size()}")

# Read a .las file
pcd = PointCloud().read_las('./data/bunny.las')
print(f"Points in bunny.las: {pcd.size()}")

# Read an .e57 file
pcd = PointCloud().read_e57('./data/bunny.e57')
print(f"Points in bunny.e57: {pcd.size()}")
```

### The `PointCloud` Class

The `open3dpypro.PointCloud` class is the main component of this extension. It wraps Open3D's `PointCloud` object and provides additional functionality.

#### Initialization

You can create a `PointCloud` object with initial data:

```python
import numpy as np
from open3dpypro import PointCloud

# Create a point cloud from numpy arrays
xyz = np.random.rand(100, 3)
rgb = np.random.rand(100, 3)
pcd = PointCloud(xyz=xyz, rgb=rgb)
```

#### File I/O

The `PointCloud` class provides methods to read from and write to various file formats:

- `read_pcd(filename)`
- `read_las(filename)`
- `read_e57(filename)`
- `save_pcd(filename)`

### Data Access

You can easily get and set point data:

- `get_points()` / `set_points(points)`
- `get_colors()` / `set_rgb(colors)`
- `get_normals()` / `set_normals(normals)`
- `get_intensity()` / `set_intensity(intensity)`

## Data

The `data` directory contains sample point cloud files:

- `bunny.pcd`
- `bunny.las`
- `bunny.e57`

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
