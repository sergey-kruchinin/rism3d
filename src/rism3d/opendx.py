import numpy as np


class openDX:
    def __init__(self, 
                 data=np.zeros((0, 0, 0)),
                 origin=np.zeros(3),
                 delta=np.zeros((3, 3))
                ):
        if not len(data.shape) == 3:
            print("WARNING: data is not 3d array")
        self.data = np.copy(data) 
        self.size = np.array(np.shape(self.data))
        if not len(origin) == 3:
            print("WARNING: origin should be 3-element array")
        self.origin = np.copy(origin)
        if not delta.shape == (3, 3):
            print("WARNING: delta should be 3x3 array")
        self.delta = np.copy(delta)

    @classmethod
    def read(cls, filename):
        size_line_token = "object 1 class gridpositions counts"
        origin_line_token = "origin"
        delta_line_token = "delta"
        data_line_token = "object 3 class array type"
        # For very old dx files produced by AmberTools 1.4
        end_line_token_old = "object \"Untitled\" call field\n"
        # For modern dx files
        end_line_token = "object \"Untitled\" class field\n"
        with open(filename) as f:
            size_string = f.readline().replace(size_line_token, "")
            size = np.array([int(i) for i in size_string.split()])
            origin_string = f.readline().replace(origin_line_token, "")
            origin = np.array([float(i) for i in origin_string.split()])
            delta = np.zeros([len(size), len(size)])
            for i in range(len(size)):
                delta_string = f.readline().replace(delta_line_token, "")
                delta[i] = [float(i) for i in  delta_string.split()] 
            f.readline()    # skip two lines 
            f.readline()
            data = np.zeros(size)
            data_flat_view = data.reshape(-1)
            flat_index = 0
            for (linenumber, line) in enumerate(f, 8):
                if line == end_line_token or line == end_line_token_old:
                    break
                try:
                    data_buffer = [float(i) for i in line.split()]
                except ValueError:
                    print("At processing line", linenumber, "in file", filename)
                    raise
                for i in data_buffer:
                    data_flat_view[flat_index] = i
                    flat_index += 1
        return cls(data, origin, delta)

    def write(self, filename):
        with open(filename, "w") as f:
            header = ("object 1 class gridpositions counts" 
                      + "".join(["{0:8d}".format(i) for i in self.size]) + "\n"
                      + "origin " 
                      + "".join(["{0:15.8f}".format(i) for i in self.origin])
                      + "\n"
                      + "delta {0:16.8f} 0 0\n".format(self.delta[0, 0])
                      + "delta  0{0:16.8f} 0\n".format(self.delta[1, 1])
                      + "delta  0 0{0:16.8f}\n".format(self.delta[2, 2])
                      + "object 2 class gridconnections counts"
                      + "".join(["{0:9d}".format(i) for i in self.size]) + "\n"
                      + "object 3 class array type double rank 0 " 
                      + "items{0:28d} data follows\n".format(self.data.size)
                     )
            f.write(header)
            lineBuffer = []
            for i in self.data.reshape(-1):
                if len(lineBuffer) == 3:
                    [f.write("{0:16.5e}".format(j)) for j in lineBuffer]
                    f.write("\n")
                    lineBuffer = []
                lineBuffer.append(i)
            if lineBuffer:
                [f.write("{0:16.5e}".format(j)) for j in lineBuffer]
                f.write("\n")
            f.write("object \"Untitled\" class field\n")

    def get_grid(self):
        delta = np.diagonal(self.delta)
        grids = [np.arange(self.size[i]) 
                 * delta[i] 
                 + self.origin[i] 
                 for i in range(3)]
        X, Y, Z = np.meshgrid(*grids, indexing="ij")
        return X, Y, Z

    def get_unit_cell_volume(self):
        delta = np.diagonal(self.delta)
        unit_cell_volume = np.prod(delta)
        return unit_cell_volume

