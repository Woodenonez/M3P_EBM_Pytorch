from typing import Union, List

import numpy as np

### Generic
class DebugTemp(): pass # temporary arguments for debugging, should delete later
class Indexable(): pass # Union[list, tuple, np.ndarray, ...]
class CoordsMatrix(Indexable):    pass # (i,j) means i-th row & j-th column
class CoordsCartesian(Indexable): pass # (x,y) means Cartesian coordinates (x,y)

### Numpy
class NumpyImageSC(np.ndarray): pass # single-channel image
class NumpyImageXC(np.ndarray): pass #  multi-channel image