'''
Helper functions and utility tools for Deep3DCCS image processing and metric calculations. 
Contains image resizing, color conversion, and file path manipulation utilities. Implements 
key evaluation metrics including Relative Percentage Error (RPE) and standard deviation 
calculations. Provides molecular visualization helpers and data formatting functions. 
Supporting module for core image processing and statistical analysis operations.
'''
import cv2
import os, sys
import time
import numpy as np 
from termcolor import colored

def get_pathname_without_ext(file_path): # Get the directory and filename without extension    
    directory = os.path.dirname(file_path)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(directory, filename)

def convert_rgb(cv2_image, bg_color =(255,255,255)):                                      # convert cv2 image convert_white_to_light_yellow
    # Define the lower and upper bounds for white color in BGR format
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # Create a mask for white pixels
    mask_white = cv2.inRange(cv2_image, lower_white, upper_white)

    
    yellow_tinted = np.zeros_like(cv2_image, dtype=np.uint8) # Create a bg_color-tinted image (light yellow)
    yellow_tinted[:] = bg_color                              # RGB values for light yellow (255, 255, 0 in OpenCV)

    # Apply the mask to select white pixels and blend with bg_color-tinted image
    result = cv2.bitwise_and(cv2_image, cv2_image, mask=~mask_white)  # Inverse mask for white pixels
    result = cv2.add(result, cv2.bitwise_and(yellow_tinted, yellow_tinted, mask=mask_white))

    return result

def original_percentage_std_error(real_values, predicted_values):
    
    real_values       = np.array(real_values) # Ensure the input arrays are numpy arrays for easier calculations
    predicted_values = np.array(predicted_values)    
    errors = real_values - predicted_values  # Calculate errors
    percentage_errors = np.abs(errors / real_values) * 100   # Calculate percentage errors (absolute errors divided by real values, then converted to %)
    std_dev_errors = np.std(errors)                   # Calculate standard deviation of errors  
    mean_percentage_error = np.mean(percentage_errors) # Calculate mean percentage error

    return mean_percentage_error, std_dev_errors

def percentage_std_error(real_values, predicted_values, threshold=100):
    """
    Calculates the mean percentage error, standard deviation of errors, and reports the count and percentage of skipped items (outliers).

    Parameters:
        real_values (list or array-like): Actual values.
        predicted_values (list or array-like): Predicted values.
        threshold (float): Maximum allowable percentage error. Values exceeding this threshold are considered outliers.

    Returns:
        tuple: mean_percentage_error, std_dev_errors, skipped_count, skipped_percentage
    """
    # Ensure inputs are 1D numpy arrays
    real_values      = np.array(real_values).flatten()
    predicted_values = np.array(predicted_values).flatten()

    # Print headers
    print("--------------[Header]--------------")
    print(f"{'Exp. CCS':<15} {'Pred. CCS':<20}")
    print("-" * 35)
    # Print aligned values
    for real, pred in zip(real_values[:10], predicted_values[:10]):
        print(colored(f"{real:<15} {pred:<20}", "yellow"))
    print(colored(".....","yellow"))
    print("-" * 35)

    # Check for consistent dimensions
    if real_values.shape != predicted_values.shape:
        raise ValueError("Error! real_values and predicted_values must have the same shape.")
    
    # Calculate errors and percentage errors
    errors = real_values - predicted_values
    percentage_errors = np.abs(errors / real_values) * 100
    
    # Identify and remove outliers based on the threshold
    mask = percentage_errors <= threshold  # Boolean mask for values within the threshold
    filtered_real_values      = real_values[mask]
    filtered_predicted_values = predicted_values[mask]
    
    # Calculate skipped items
    total_count = len(real_values)
    skipped_count = total_count - np.sum(mask)  # Total items minus retained items
    skipped_percentage = (skipped_count / total_count) * 100 if total_count > 0 else 0
    skipped_percentage = round(skipped_percentage,4)
    
    # Recalculate errors and percentage errors after removing outliers
    filtered_errors = filtered_real_values - filtered_predicted_values
    filtered_percentage_errors = np.abs(filtered_errors / filtered_real_values) * 100
    
    # Calculate metrics for relative percentage error
    mean_percentage_error = np.mean(filtered_percentage_errors)  # average relative percentage error
    std_dev_errors        = np.std(filtered_percentage_errors)   # std within average percentage error
    
    return mean_percentage_error, std_dev_errors, skipped_count, skipped_percentage 



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):  # takes Cv2 image and resize based in height ratio
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


if __name__ == "__main__":
    # just test
    img =cv2.imread("./assets/sample_projection.png") 
    img =convert_rgb(img , bg_color =  (180,180,180))   
    cv2.imshow("tite", img)
    cv2.waitKey(0)