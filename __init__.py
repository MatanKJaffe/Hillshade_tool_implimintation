from read_raster import ReadRaster
def main():
    infile = "C:/Users/matan/OneDrive/Documents/img_test/image_base_archive/test1.tiff"
    outfile = "for moran/hillshade_test1.tiff"
    ReadRaster(infile, outfile).process_windows(windowing_type = None)
if __name__ == "__main__":main()