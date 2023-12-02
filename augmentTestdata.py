import cv2
import argparse
import sys
import os
from auxilary.utils import createDir
from natsort import natsorted

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='none', help='Path to the experiment directory.')
    return parser.parse_args()

def crop_image(image_path, output_folder, counter, crop_size):

    img = cv2.imread(image_path)
    label = cv2.imread(image_path.replace(".png", "_label.png"))
    height, width = img.shape[:2]

    # Calculate crop dimensions
    crop_width, crop_height = crop_size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = (width + crop_width) // 2
    bottom = (height + crop_height) // 2

    # Center crop
    center_crop_img = img[top:bottom, left:right]
    cv2.imwrite(f'{output_folder}{counter}.png', center_crop_img)
    center_crop_label = label[top:bottom, left:right]
    cv2.imwrite(f'{output_folder}{counter}_label.png', center_crop_label)
    counter += 1


    # Top-left crop
    top_left_crop_img = img[0:crop_height, 0:crop_width]
    cv2.imwrite(f'{output_folder}{counter}.png', top_left_crop_img)
    top_left_crop_label = label[0:crop_height, 0:crop_width]
    cv2.imwrite(f'{output_folder}{counter}_label.png', top_left_crop_label)
    counter += 1

    # Top-right crop
    top_right_crop_img = img[0:crop_height, width-crop_width:width]
    cv2.imwrite(f'{output_folder}{counter}.png', top_right_crop_img)
    top_right_crop_label = label[0:crop_height, width-crop_width:width]
    cv2.imwrite(f'{output_folder}{counter}_label.png', top_right_crop_label)
    counter += 1
    

    # Bottom-left crop
    bottom_left_crop_img = img[height-crop_height:height, 0:crop_width]
    cv2.imwrite(f'{output_folder}{counter}.png', bottom_left_crop_img)
    bottom_left_crop_label = label[height-crop_height:height, 0:crop_width]
    cv2.imwrite(f'{output_folder}{counter}_label.png', bottom_left_crop_label)
    counter += 1

    # Bottom-right crop
    bottom_right_crop_img = img[height-crop_height:height, width-crop_width:width]
    cv2.imwrite(f'{output_folder}{counter}.png', bottom_right_crop_img)
    bottom_right_crop_label = label[height-crop_height:height, width-crop_width:width]
    cv2.imwrite(f'{output_folder}{counter}_label.png', bottom_right_crop_label)
    counter += 1
    

    return counter

# Example usage
# crop_image('path_to_your_image.jpg', 'path_to_output_folder', (200, 200))

if __name__ == "__main__":

    args = arg_init()
    if args.img_dir == 'none':
        print("Please provide the test image directory")
        sys.exit()

    outdir = args.img_dir+"cropped/"
    createDir([outdir])

    imgCounter = 0

    for i in range(len(os.listdir(args.img_dir))//2):
        fname = f"{i}.png"
        imgCounter = crop_image(args.img_dir+fname, outdir, imgCounter, (400,400))
