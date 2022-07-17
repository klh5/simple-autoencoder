from osgeo import gdal
import numpy as np
import glob
import os
from pathlib import Path
from matplotlib import pyplot as plt
import argparse

def pad_image(img, tile_size):
    '''Pads the image so that it is a multiple of the tile size.'''
    
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    pad_height = tile_size - (img_height - (tile_size * (img_height // tile_size)))
    pad_width = tile_size - (img_width - (tile_size * (img_width // tile_size)))
    
    padded_img = np.pad(img, ((0, pad_height), (0, pad_width)))
    
    return(padded_img)

def tile_image(img, tile_size):
    '''Splits the image into tiles of the correct size.'''
    
    vtiles = int(img.shape[0]/tile_size) # Number of tiles vertically
    htiles = int(img.shape[1]/tile_size) # Number of tiles horizontally
    
    tiles = img.reshape(vtiles, tile_size, htiles, tile_size, 1)
    tile_arr = tiles.swapaxes(1, 2)
    tile_stack = tile_arr.reshape(-1, tile_size, tile_size)
    
    return(tile_stack)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Split images from a directory into tiles.')
    
    parser.add_argument('-ts', '--tile_size', default=64, type=int, required=True, help="Size of required tiles where height = width.")
    parser.add_argument('-d', '--img_dir', required=True, help="The directory containing input images.")
    parser.add_argument('-p', '--file_pattern', required=True, help="Expression used with glob to match input files.")
    parser.add_argument('-o', '--output_dir', required=True, help="The output directory where the tiles will be stored.")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    path = Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Get list of input images from directory
    img_list = glob.glob(args.img_dir + args.file_pattern)
    
    print("Found {} images in {}".format(len(img_list), args.img_dir))
    
    # For every image in the directory
    for dune_image in img_list:

        try:
            img = gdal.Open(dune_image)
        except:
            print("Could not load image: {}".format(dune_image))
            continue
        
        # Convert image to Numpy array
        band = img.GetRasterBand(1)
        img_data = band.ReadAsArray()
        
        # Change masked out regions to 0 as nodata is quite high
        # This is assuming the input is a tif (uint16)
        img_data[img_data == 65535] = 0
    
        # Pad and tile
        padded_img = pad_image(img_data, args.tile_size)
        tile_stack = tile_image(padded_img, args.tile_size)

        num_tiles = tile_stack.shape[0]

        print("Generated {} tiles from {}".format(num_tiles, dune_image))

        # Get filename without extension
        image_base = os.path.basename(dune_image).split('.')[0]

        # Create output directory
        image_output_dir = os.path.join(args.output_dir, image_base)

        # Include tile size in output path, so we don't overwrite tiles
        image_output_tile_size = os.path.join(image_output_dir, str(args.tile_size))

        # Create output directory if it doesn't already exist
        path = Path(image_output_tile_size)
        path.mkdir(parents=True, exist_ok=True)

        # Save numpy array stack
        data_stack_name = os.path.join(image_output_tile_size, 'tile_data.npy')
        np.save(data_stack_name, tile_stack)
        
        # For each tile, save it as a separate image
        for i in range(num_tiles):
            tile = tile_stack[i] # Subset tile

            # Save png image
            tile_name = 'tile_{}.png'.format(i)
            full_tile_path = os.path.join(image_output_tile_size, tile_name)
            plt.imsave(full_tile_path, tile)