"""thread_pool_executor.py

Operate on a raster dataset window-by-window using a ThreadPoolExecutor or a Pro

With -j 4, the program returns in about 1/4 the time as with -j 1.
"""

import concurrent.futures
import threading
from itertools import product
import rasterio as rio
from rasterio import windows
import numpy as np
from numba import jit
from numpy import gradient, ndarray, uint8 ,pi ,arctan ,arctan2 , sin, cos, sqrt, degrees, array as nparray
from dsm_funcs import DSM_FUNCS

class ReadRaster:
    """
    _summary_
    """
    def __init__(self,src_img, dst_img = None):#, **func_kwargs):
        self.src_img= src_img
        self.dst_img = dst_img 
        #self.func = func
        #self.func_kwargs = func_kwargs
   
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

    def compute(self, array):
        return self.hillshade(array)
        #return self.func(array)

    def _read_process_write_with_threadpool(self, src, dst,windows, num_workers:int=400):
        """Process infile block-by-block and write to a new file.
        """
        # We cannot write to the same file from multiple threads
        # without causing race conditions. To safely read/write
        # from multiple threads, we use a lock to protect the
        # DatasetReader/Writer
        read_lock = threading.Lock()
        write_lock = threading.Lock()
        def process(window):
            with read_lock:
                src_array = src.read(window=window).astype(uint8)
   
            # The computation can be performed concurrently
            result = nparray((self.hillshade(band) for band in src_array), dtype = uint8)

            with write_lock:
                dst.write(result, window=window)
        
        # We map the process() function over the list of
        # windows.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            executor.map(process, windows)

    def process_windows(self, Radius_px_lst:list = [], overlap:int = 1, windowing_type:str = "windows", boundless:bool = False, with_threads:bool = False):
        """width & height not including overlap i.e requesting a 256x256 window with 
            1px overlap will return a 258x258 window (for non edge windows)
        """        
        with rio.Env(CPL_DEBUG=True, GDAL_CACHEMAX=1073741824):
            # This ensures that all drivers are registered in the global
            # context. Within this block *only* GDAL's debugging messages
            # are turned on and the raster block cache size is set to 1 gb.
            with rio.open(self.src_img,'r',format='GTiff') as src:
                if self.dst_img is not None:
                    kwargs = src.profile
                    kwargs.update(dtype=rio.int32,count=3, blockxsize=256, blockysize=256, tiled=True)
                    with rio.open(self.dst_img, 'w', **kwargs) as dst:
                        #with rio.open(self.dst_img, 'w') as dst:
                        if windowing_type == "windows":

                            if any(np.array(src.shape) < np.array([256,256])):
                                windowing_type = None
                            else:
                                if with_threads is True:
                                        self._read_process_write_with_threadpool(src, dst, [window for window in self._get_overlapping_windows(src, boundless, overlap)])
                                else:
                                    for window in self._get_overlapping_windows(src, boundless, overlap):
                                    
                                        for band_id in src.indexes:
                                            self._read_process_write(src, dst, band_id, window = window)
                            
                        if (windowing_type == "blocks") | (windowing_type == "tiles"):
                            if not any(np.array(src.shape) > np.array(set(x for i in set(src.block_shapes) for x in i))):
                                windowing_type = None
                            else:
                                if with_threads is True:
                                        self._read_process_write_with_threadpool(src, dst, [window for window in self._get_overlapping_blocks(src, boundless, overlap)])
                                else:
                                    for band_id in src.indexes:
                                        for window in self._get_overlapping_blocks(src, boundless,  overlap, band_id):
                                            self._read_process_write(src, dst, band_id, Radius_px_lst, window = window)
                        
                        if windowing_type is None:
                            for band_id in src.indexes:
                                self._read_process_write(src, dst, band_id, Radius_px_lst)
                        
                        #if windowing_type == "simple":
                        ##    #out = np.zeros(src.shape)
                        #    #for band_id in src.indexes:
                        #    arr = src.read(1) + src.read(2) + src.read(3)
                        #    dst_data  = AnomalyIndex().execute(arr,Radius_px_lst)# Do the Processing Here\
                        #    #x,y = np.indices(arr.shape)
                        #    dst.write(dst_data, 1)
                        
    def _read_process_write(self, src, dst, band_id:int, window = None):
        src_data = src.read(band_id, window = window)
        #obj = self.obj
        #func = self.func
        #dst_data = obj.func(src_data, self.func_kwargs)
        dst_data = self.hillshade(src_data)# Do the Processing Here
        dst.write(dst_data, band_id, window = window)
    
    def _get_overlapping_windows(self, src, boundless:bool,  overlap:int=0, width:int=256, height:int=256):
        """"width & height not including overlap i.e requesting a 256x256 window with 
            1px overlap will return a 258x258 window (for non edge windows)"""
        #if window size is larger than array, switch to simple reader 
        offsets = product(range(0, src.meta['width'], width), range(0, src.meta['height'], height))
        big_window = windows.Window(col_off=0, row_off=0, width=src.meta['width'], height=src.meta['height'])
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off - overlap, row_off=row_off - overlap, width=width + overlap * 2, height=height + overlap * 2)
            if boundless:
                yield window

            else:
                yield window.intersection(big_window)

    def _get_overlapping_blocks(self,src, boundless:bool, overlap:int=0, band:int=1):
        # if block size is smaller than the max radius, try to chamge the block size/ switch to window reader
        big_window = windows.Window(col_off=0, row_off=0, width=src.meta['width'], height=src.meta['height'])
        for ji, window in src.block_windows(band):
            if overlap == 0:
                yield window
            else:
                window = windows.Window(
                    col_off=window.col_off - overlap,
                    row_off=window.row_off - overlap,
                    width=window.width + overlap * 2,
                    height=window.height + overlap * 2)
                if boundless:
                    yield window
                else:
                    yield window.intersection(big_window)