import cv2
import numpy as np
import argparse
import sys
import os

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_dir', type=str, default='none', help='Path to the experiment directory.')
    return parser.parse_args()


def post_process(model_output, min_area=100):
    """
    Post-processes the model output to remove tiny objects and fill holes.
    
    Parameters:
    - model_output: np.ndarray, binary image output from the model
    - min_area: int, minimum area of the object to be considered valid
    
    Returns:
    - post_processed_img: np.ndarray, post-processed image
    """
    
    # Step 1: Remove small objects
    contours, _ = cv2.findContours(model_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            cv2.drawContours(model_output, [cnt], 0, 0, -1)
    
    # Step 2: Fill holes
    kernel = np.ones((5, 5), np.uint8)
    post_processed_img = cv2.morphologyEx(model_output, cv2.MORPH_CLOSE, kernel)
    
    return post_processed_img

def dice_score(image1, image2, threshold=128):
    """
    Calculate the Dice score, a measure of set similarity.
    
    Parameters:
    image1 (numpy.ndarray): The first image (binary).
    image2 (numpy.ndarray): The second image (binary).
    
    Returns:
    float: Dice score ranging from 0 to 1. A score of 1 denotes perfect similarity.
    """
    if image1.shape != image2.shape:
        raise ValueError("Shape mismatch: image1 and image2 must have the same shape.")
    

    # Convert grayscale images to binary
    binary1 = (image1 >= threshold).astype(np.uint8)
    binary2 = (image2 >= threshold).astype(np.uint8)

    # Calculate intersection and union
    intersection = np.logical_and(binary1, binary2)
    union = np.logical_or(binary1, binary2)

    # Calculate Dice score
    dice = 2. * intersection.sum() / (binary1.sum() + binary2.sum())

    return dice


if __name__ == "__main__":
    args = arg_init()

    if args.expt_dir == 'none':
        print("Please specify experiment directory")
        sys.exit(1)

    predicted_img_dir = args.expt_dir

    dices = []

    for filename in os.listdir(predicted_img_dir):
        
        if filename.endswith('predict.png'): 
            file_path = os.path.join(predicted_img_dir, filename)
            predicted_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if predicted_img is not None:
                post_processed_img = post_process(predicted_img)
                labelName = file_path.split("/")[-1]
                label_imgPath = labelName.split("_")[0]+"_label.png"
                
                print(label_imgPath)
                label_img = cv2.imread(predicted_img_dir+label_imgPath, cv2.IMREAD_GRAYSCALE)

                dice = dice_score(label_img, post_processed_img)
                dices.append(dice)

                # Save the post-processed image
                post_processed_img_path = os.path.join(predicted_img_dir,'post_' +str(dice)+"_"+ filename)
                print(post_processed_img_path)
                #cv2.imwrite(post_processed_img_path, post_processed_img)
            else:
                print(f"Failed to read image: {file_path}")

    print(f"Average Dice Score: {np.average(dices)}")