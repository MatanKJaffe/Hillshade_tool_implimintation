from numba import jit
from numpy import gradient, ndarray, uint8 ,pi ,arctan ,arctan2 , sin, cos, sqrt, degrees, array as nparray

class DSM_FUNCS:  
    def __init__(self,function:str = "hillshade"):
        if function not in ["hillshade", "slope", "aspect"]:
            raise ValueError("please enter a valid function name from the list : hillshade, slope, aspect")
        self.function = function

    @jit(forceobj=True)
    def hillshade(self, array:ndarray ,azimuth:int = 315 ,angle_altitude:int = 45) -> ndarray:
        azimuth = 360.0 - azimuth 
        x, y = gradient(array)
        slope = pi/2. - arctan(sqrt(x*x + y*y))
        aspect = arctan2(-x, y)
        azimuthrad = azimuth*pi/180.
        altituderad = angle_altitude*pi/180.
        shaded = sin(altituderad)*sin(slope) + cos(altituderad)*cos(slope)*cos((azimuthrad - pi/2.) - aspect)
        return nparray(255*(shaded + 1)/2 , dtype = uint8) 

    @jit(forceobj=True)
    def aspect(self, array:ndarray, deg:bool = True) -> ndarray:  
        x, y = gradient(array)
        if deg is True:
            return nparray(degrees(arctan2(-x, y)), dtype = uint8)
        return  nparray(arctan2(-x, y), dtype = uint8)
        
    @jit(forceobj=True)
    def slope(self, array:ndarray, deg:bool = True) -> ndarray:
        x, y = gradient(array)
        if deg is True:
            return  nparray(degrees(pi/2. - arctan(sqrt(x*x + y*y))), dtype = uint8)
        return nparray(pi/2. - arctan(sqrt(x*x + y*y)), dtype = uint8)

    def execute(self, array):
        if self.function == "hillshade":
            return self.hillshade(array)
        elif self.function == "slope":
            return self.slope(array)
        elif self.function == "aspect":
            return self.aspect(array)
        raise ValueError("please enter a valid function name from the list : hillshade, slope, aspect")

