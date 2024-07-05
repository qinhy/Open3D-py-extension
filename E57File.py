import numpy as np
import pye57
from pye57.e57 import SUPPORTED_POINT_FIELDS
import re

SUPPORTED_POINT_FIELDS["nor:normalX"] = "f"
SUPPORTED_POINT_FIELDS["nor:normalY"] = "f"
SUPPORTED_POINT_FIELDS["nor:normalZ"] = "f"

class E57File:
    def __init__(self,point_file,intensity=False, colors=False, row_column=False, printinfo=False) -> None:
        self.intensity = intensity
        self.colors = colors
        self.row_column = row_column
        self.row_index = row_column
        self.column_index = row_column
        self.point_file = point_file
        self.e57 = pye57.E57(point_file)
        self.close = self.e57.close
        def read_scan_raw_gen(index, every_k_points=10000000):
            header = self.e57.get_header(index)    
            def make_buffers(header,k_points):
                data = {}
                buffers = pye57.libe57.VectorSourceDestBuffer()
                for field in header.point_fields:
                    if field not in SUPPORTED_POINT_FIELDS:continue
                    np_array, buffer = self.e57.make_buffer(field, k_points)
                    data[field] = np_array
                    buffers.append(buffer)
                return data,buffers    
            hpc = header.point_count
            ekp = every_k_points    
            
            data,buffers = make_buffers(header,every_k_points)
            reader = header.points.reader(buffers)
            
            for i in list(range(0,hpc//ekp*ekp,ekp))+[hpc]:
                k_points=ekp if i%ekp==0 else i%ekp
                reader.read()
                yield {k:v[:k_points].copy() for k,v in data.items()}       
                
        setattr(self.e57, 'read_scan_raw_gen', read_scan_raw_gen)
        
        self.scan_count = self.e57.scan_count
        self.scan_headers = [self.e57.get_header(i) for i in range(self.scan_count)]
        self.scan_headers_dict = [self.parse_structure_node('\n'.join(h.pretty_print())) for h in self.scan_headers]
        self.scan_cartesianMinBounds = np.asarray([(h.cartesianBounds['xMinimum'].value(),
                                                      h.cartesianBounds['yMinimum'].value(),
                                                      h.cartesianBounds['zMinimum'].value(),
                                                      ) for h in self.scan_headers])
        
        self.scan_cartesianMaxBounds = np.asarray([(h.cartesianBounds['xMaximum'].value(),
                                                      h.cartesianBounds['yMaximum'].value(),
                                                      h.cartesianBounds['zMaximum'].value(),
                                                      ) for h in self.scan_headers])
        
        self.scan_headers_has_intensity = ['intensity' in h.point_fields for h in self.scan_headers]
        self.scan_headers_has_color = ['colorRed' in h.point_fields for h in self.scan_headers]
        self.scan_headers_has_row_column = [('rowIndex' in h.point_fields and 'columnIndex' in h.point_fields) for h in self.scan_headers]

        self.cartesianMinBound = self.scan_cartesianMinBounds.min(0)
        self.cartesianMaxBound = self.scan_cartesianMaxBounds.max(0)
        
        self.point_counts = [i.point_count for i in self.scan_headers]
        self.all_points = sum(self.point_counts)
        if printinfo:            
            print('[E57File]','point size :', self.all_points,'scans :', self.scan_count)
            print('[E57File]','each scan point size',list(zip(range(self.scan_count),self.point_counts)))
        self.datagen = self.gen()
    
    def has_intensity(self):return all(self.scan_headers_has_intensity)
    def has_color(self):return all(self.scan_headers_has_color)
    def has_row_column(self):return all(self.scan_headers_has_row_column)

    def parse_structure_node(self,s):
        res = {}
        for i in re.split('StructureNode([\n\s\S]+?\n)<', s):
            if len(i)<10:continue
            key = re.search(' \'(.+?)\'>\n', i)
            if key is not None:            
                key = key.group().split("'")[1]
                i = i.replace(f"StructureNode '{key}'>\n",'')
                i = i.replace(f"'{key}'>\n",'')
                i = i.replace('    <','<')
                if 'StructureNode' in i:
                    res.update({key:self.parse_structure_node(i)})
                else:
                    res.update({key:self.parse_node(i)})
            else:
                res.update(self.parse_node(i))
        return res

    def parse_node(self,data):
        result = {}
        lines = data.split('\n')
        current_dict = result
        dict_stack = []
        for line in lines:
            match = re.match(r"<(\w+)Node '(\w+)'>: (.+)", line.strip())
            if match:
                node_type, key, value = match.groups()
                if node_type in ['String', 'Integer', 'Float']:
                    if node_type == 'String':
                        value = value#[1:-1]  # Remove quotes around string
                    elif node_type == 'Integer':
                        value = int(value)
                    elif node_type == 'Float':
                        value = float(value)
                    current_dict[key] = value
            else:
                structure_match = re.match(r"<(\w+)Node '(\w+)'>", line.strip())
                if structure_match:
                    node_type, key = structure_match.groups()
                    new_dict = {}
                    current_dict[key] = new_dict
                    dict_stack.append(current_dict)
                    current_dict = new_dict
                elif line.strip() == '}':
                    current_dict = dict_stack.pop()
        return result
    
    def e57assert(self,data,intensity,colors,row_column=False):
        assert isinstance(data["cartesianX"], np.ndarray)
        assert isinstance(data["cartesianY"], np.ndarray)
        assert isinstance(data["cartesianZ"], np.ndarray)
        if intensity:
            assert isinstance(data["intensity"], np.ndarray),'the file has no intensity'
        if colors:
            assert isinstance(data["colorRed"], np.ndarray),'the file has no color'
            assert isinstance(data["colorGreen"], np.ndarray)
            assert isinstance(data["colorBlue"], np.ndarray)
        if row_column:
            assert isinstance(data["rowIndex"], np.ndarray),'the file has no row and column index'
            assert isinstance(data["columnIndex"], np.ndarray)

    def readall(self):
        xyzs,rgbs,intensitys = [],[],[]
        row_index,column_index = [],[]
        for i in range(self.e57.scan_count):
            xyz,rgb,inten,row,col = self.read(i)
            xyzs.append(xyz)
            rgbs.append(rgb)
            intensitys.append(inten)
            row_index.append(row)
            column_index.append(col)
        return np.vstack(xyzs),np.vstack(rgbs),np.hstack(intensitys),np.hstack(row_index),np.hstack(column_index)
    
    def _set_data_to_numpy(self,data):
        self.xyz = np.stack([data[i] for i in ['cartesianX','cartesianY','cartesianZ']], axis=1)
        try :
            self.rgb = np.stack([data[i] for i in ['colorRed','colorGreen','colorBlue']], axis=1) if self.colors else []
        except Exception as e:
            print('[E57File@data_to_numpy@rgb]',e)
            self.rgb = []

        try :
            self.inten = np.array(data["intensity"]) if self.intensity else []
        except Exception as e:
            print('[E57File@data_to_numpy@inten]',e)
            self.inten = []

        try :
            self.row_index = np.array(data["rowIndex"]) if self.row_column else []
        except Exception as e:
            print('[E57File@data_to_numpy@row_index]',e)
            self.row_index = []

        try :
            self.column_index = np.array(data["columnIndex"]) if self.row_column else []
        except Exception as e:
            print('[E57File@data_to_numpy@column_index]',e)
            self.column_index = []

    def read(self,idx):
        print('[E57File@read] reading file :',self.point_file,', idx :',idx)
        data = self.e57.read_scan(idx,ignore_missing_fields=True, intensity=self.intensity, colors=self.colors, row_column=self.row_column)
        self.e57assert(data,intensity=self.intensity, colors=self.colors, row_column=self.row_column)
        self._set_data_to_numpy(data)
        print(f'[E57File@read] reading scan : No.{idx}, points :{len(self.xyz)}, rgb :{len(self.rgb)}, intensity :{len(self.inten)}, '+
                f'row_index :{len(self.row_index)}, column_index :{len(self.column_index)}')          
        return self.xyz,self.rgb,self.inten,self.row_index,self.column_index

    def read_scan_gen(self, idx, every_k_points=10000000):
        for data_No,data in enumerate(self.e57.read_scan_raw_gen(idx, every_k_points)):
            self.e57assert(data,intensity=self.intensity, colors=self.colors, row_column=self.row_column)
            self._set_data_to_numpy(data)
            print(f'[E57File@read] reading scan : No.{idx}, ChunckNo.{data_No}, points :{len(self.xyz)}, rgb :{len(self.rgb)}, intensity :{len(self.inten)}, '+
                  f'row_index :{len(self.row_index)}, column_index :{len(self.column_index)}')
            yield self.xyz,self.rgb,self.inten,self.row_index,self.column_index

    def gen(self):
        for i in range(self.e57.scan_count):
            self.read(i)
            yield self.xyz,self.rgb,self.inten,self.row_index,self.column_index

    def __next__(self):
        return next(self.datagen)

