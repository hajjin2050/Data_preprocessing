import cv2
import numpy as np

def calculate_exg(image):
    # Split the image into BGR channels
    
    b, g, r = cv2.split(image)

    # Calculate ExG
    # Note: `astype` is used to prevent overflow
    exg = 2.8*g.astype(float) - 1.8*r.astype(float) - 0.9*b.astype(float)
    exg = cv2.normalize(exg, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    return exg


def save_image(image, output_path):
    cv2.imwrite(output_path, image)

from osgeo import gdal, osr

def remove_soil_color(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for soil color
    lower = np.array([0, 100, 50])
    upper = np.array([120, 100, 50])

    # Create a mask for the soil color
    mask = cv2.inRange(hsv, lower, upper)

    # Apply the mask to the image
    image[mask > 0] = [0, 0, 0]

    return image

def save_image_with_geoinfo(input_path, output_image, output_path):
    # Open the input image and get its geotransform and projection
    input_ds = gdal.Open(input_path)
    geotransform = input_ds.GetGeoTransform()
    projection = input_ds.GetProjection()
    input_ds = None  # Close the dataset

    # Normalize the output image to [0, 1]
  
    # Create the output dataset
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_path, output_image.shape[1], output_image.shape[0], 1, gdal.GDT_Float32)

    # Set the geotransform and projection
    output_ds.SetGeoTransform(geotransform)
    output_ds.SetProjection(projection)

    # Write the output image
    output_ds.GetRasterBand(1).WriteArray(output_image)
    output_ds = None  # Close the dataset
    
def preprocess_image_with_exr(input_path, output_path):
    # Load the input image
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    # Remove soil color from the image
    image = remove_soil_color(image)

    # Calculate ExR
    exr = calculate_exg(image)

    # Threshold the ExR image to get a binary mask
    _, binary_mask = cv2.threshold(exr, 150, 255, cv2.THRESH_BINARY)

    # Normalize the binary mask to [0, 1]
    binary_mask = binary_mask / 255.0

    # Save the binary mask with the geoinformation of the input image
    save_image_with_geoinfo(input_path, binary_mask, output_path)

import os
import numpy as np

def remove_images_with_low_green_pixels(image_folder, mask_folder, threshold=0.9):
    # Get the list of image files
    image_files = os.listdir(image_folder)

    for image_file in image_files:
        # Construct the full path of the image file and the corresponding mask file
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, image_file)

        # Preprocess the image and save the result
        preprocess_image_with_exr(image_path, 'temp.tif')

        # Load the preprocessed image
        preprocessed_image = cv2.imread('temp.tif', cv2.IMREAD_GRAYSCALE)

        # Calculate the ratio of green pixels
        green_ratio = np.count_nonzero(preprocessed_image) / (preprocessed_image.shape[0] * preprocessed_image.shape[1])

        # If the ratio of green pixels is below the threshold, remove the image and the mask
        if green_ratio < threshold:
            os.remove(image_path)
            os.remove(mask_path)

# Call the function
remove_images_with_low_green_pixels('path_to_image_folder', 'path_to_mask_folder')
