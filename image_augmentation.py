import math
import cv2
import numpy as np

from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import os
import skimage.io
import glob

from auxilary.utils import toGray, toGray4C, readConfig
import logging
from tqdm import tqdm
from natsort import natsorted

np.random.seed(seed=2020)

### from: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)
    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST ## GROUND TRUTH는INTER_NEAREST로 해야함.
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    scaleFactor = 0.5
    #scaleFactor = 0.8

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * scaleFactor)
    x2 = int(image_center[0] + width * scaleFactor)
    y1 = int(image_center[1] - height * scaleFactor)
    y2 = int(image_center[1] + height * scaleFactor)

    return image[y1:y2, x1:x2]

### from: https://www.kaggle.com/safavieh/image-augmentation-using-skimage
def randRange(a, b):
    return np.random.rand() * (b - a) + a

def randomIntensity(im):
    # rescales the intesity of the image to random interval of image intensity distribution
    return rescale_intensity(im,
                             in_range=tuple(np.percentile(im, (randRange(0, 10), randRange(90, 100)))),
                             out_range=tuple(np.percentile(im, (randRange(0, 10), randRange(90, 100)))))

def randomGamma(im):
    # Gamma filter for contrast adjustment with random gamma value.
    return adjust_gamma(im, gamma=randRange(1, 2.5))

def randomGaussian(im):
    # Gaussian filter for bluring the image with random variance.
    return gaussian(im, sigma=randRange(0, 2))

def randomNoise(im):
    # random gaussian noise with random variance.
    var = randRange(0.0009, 0.004)
    return random_noise(im, var=var)

def randomFilter(img, prob):
    '''
    기존 필터   : equalize_adapthist, equalize_hist,
    수정된 필터 : randomIntensity(랜덤으로 변경됨),
    추가된 필터 : randomGamma, randomGaussian,
    probability는 uniform 분포를 따름.
    '''
    img = img.astype(np.float32)
    img /= 255.
    if prob < 0.1:  # 10%
        return equalize_adapthist(img), 'eqada'
    elif prob < 0.2:  # 10%
        return equalize_hist(img), 'eqhist'
    elif prob < 0.3:  # 10%
        return randomGamma(img), 'gamma'
    elif prob < 0.5:  # 20%
        #return randomGaussian(img), 'gauss'
        return img, 'origin'
    elif prob < 0.7:  # 20%
        return randomIntensity(img), 'inten'
    elif prob < 0.8:
        return randomNoise(img), 'noise'
    else:  # 30%
        return img, 'origin'

def flip_lr(img, prob):
    if prob <= 0.5:
        return img, 'F'
    else:
        return np.fliplr(img), 'T'

def rotate_crop_radom(img, angle):
    image_height, image_width = img.shape[0:2]

    image_rotated = rotate_image(img, angle)
    image_rotated_cropped = crop_around_center(image_rotated,
                                               *largest_rotated_rect(image_width, image_height, math.radians(angle)))

    return image_rotated_cropped, str(angle)


if __name__ == '__main__':

    # 이미지 하나당 얼마나 늘릴 껀지.

    config = readConfig()
    log_dir = config["log"]
    f = open(log_dir +  "augmentationLog.txt", "w")

    augment_num_per_img = config["augmentPerImage"]
    tile_width = config["finalTileWidth"]
    tile_height = config["finalTileHeight"]

    slidingDir = config["out_dir"]

    labeled_imgs = natsorted(glob.glob(f'{slidingDir}labels/*'))
    #boundary_imgs = natsorted(glob.glob(f'{slidingDir}edge/*'))
    raw_imgs = natsorted(glob.glob(f'{slidingDir}images/*'))

    """ labeled_imgs = sorted(glob.glob('../data/data_original/label/*'))
    raw_imgs = sorted(glob.glob('../data/data_original/original/*')) """
    print(len(raw_imgs) == len(labeled_imgs))

    logging.basicConfig(level=logging.CRITICAL)

    base_dir = config["augmented_dir"]
    new_raw_dir = f'{base_dir}original/'
    new_labeled_dir = f'{base_dir}label/'
    #new_boundary_dir = f'{base_dir}edge/'

    for d in [base_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    count = 0
    #fops = open("log.txt", "w")
    #fops.write(raw_imgs)
    """ print(raw_imgs[:20])
    print(labeled_imgs[:20])
    exit(0) """
    for i in tqdm(range(len(labeled_imgs))):

        #count+=1
        #print("Image -"+str(i))
        
        raw_img = np.array(skimage.io.imread(raw_imgs[i]))
        labeled_img = np.array(skimage.io.imread(labeled_imgs[i]))
        #boundary_img = np.array(skimage.io.imread(boundary_imgs[i]))
        


        logging.debug("size1:" + str(labeled_img.shape))

        raw_name = raw_imgs[i].split('/')[-1].replace(' ', '')
        labeled_name = labeled_imgs[i].split('/')[-1].replace(' ', '')
        #boundary_name = boundary_imgs[i].split('/')[-1].replace(' ', '')

        true_raw = raw_img
        true_label = labeled_img
        #true_boundary = boundary_img

        # random numbers
        flip_random = np.random.uniform(size=augment_num_per_img)  # [0,1]

        rotate_radom = np.random.randint(low=0, high=360, size=augment_num_per_img)  # [0, 360]

        resize_x = np.random.normal(loc=0, scale=0.04, size=augment_num_per_img)
        resize_y = np.random.normal(loc=0, scale=0.04, size=augment_num_per_img)

        color_modification_random = np.random.uniform(size=augment_num_per_img)

        modified = ''
        
        original_raw_img = np.copy(raw_img)
        original_label_img = np.copy(labeled_img)
        #original_boundary_img = np.copy(boundary_img)

        origFLag = True

         
        for j in range(augment_num_per_img):

            # flip randlomly
            raw_img, flag = flip_lr(original_raw_img, flip_random[j])
            labeled_img, flag = flip_lr(original_label_img, flip_random[j])
            #boundary_img, flag = flip_lr(original_boundary_img, flip_random[j])

            modified = flag

            # rotate randomly without blank
            raw_img, flag = rotate_crop_radom(raw_img, rotate_radom[j])
            labeled_img, flag = rotate_crop_radom(labeled_img, rotate_radom[j])
            #boundary_img, flag = rotate_crop_radom(boundary_img, rotate_radom[j])

            modified += '_' + flag

            # resize randomly: labeled는  INTER NEAREST가 필수이지만, raw_img는 필수(최적)은 아닐 수 있다.
            raw_img = cv2.resize(raw_img, dsize=(0, 0), fx=(1 + resize_x[j]), fy=(1 + resize_y[j]),
                                 interpolation=cv2.INTER_NEAREST)
            labeled_img = cv2.resize(labeled_img, dsize=(0, 0), fx=(1 + resize_x[j]), fy=(1 + resize_y[j]),
                                     interpolation=cv2.INTER_NEAREST)
            #boundary_img = cv2.resize(boundary_img, dsize=(0, 0), fx=(1 + resize_x[j]), fy=(1 + resize_y[j]),
                                     #interpolation=cv2.INTER_NEAREST)
            
            #labeled_img = toGray(cv2.cvtColor(labeled_img, cv2.COLOR_RGB2GRAY))
            #boundary_img = toGray4C(cv2.cvtColor(boundary_img, cv2.COLOR_RGB2GRAY))

            modified += '_' + str(1 + resize_x[j])[2:4] + '_' + str(1 + resize_y[j])[2:4]

            # color modification. 255로 다시 곱해줘야하는지 확인.
            raw_img, mod = randomFilter(raw_img, color_modification_random[j])

            modified += '_' + mod + '_'

            # save image, file type should be in 3 letter 여야함.
            new_raw_name = str(count) + raw_name[-4:]
            new_labeled_name = str(count) + "_label" + labeled_name[-4:]
            #new_boundary_name = str(count) + "_label_b" + labeled_name[-4:]

            raw_img = np.clip(raw_img, -1.0, 1.0)  #
            raw_img = np.uint8((raw_img + 1.0) * 127.5)  # [-1,1] -> [0,255]

            #save original image
            """ if origFLag:
                skimage.io.imsave(base_dir + new_raw_name, true_raw[0:tile_height,0:tile_width])
                skimage.io.imsave(base_dir + new_labeled_name, toGray(cv2.cvtColor(true_label[0:tile_height,0:tile_width], cv2.COLOR_BGR2GRAY)))
                count+=1
                origFLag = False """

            new_raw_name = str(count) + raw_name[-4:]
            new_labeled_name = str(count) + "_label" + labeled_name[-4:]
            #print(raw_img[0:tile_height,0:tile_width].shape)
            #print(labeled_img[0:tile_height,0:tile_width].shape)
            cv2.imwrite(base_dir + new_raw_name, raw_img[0:tile_height,0:tile_width])
            cv2.imwrite(base_dir + new_labeled_name, labeled_img[0:tile_height,0:tile_width])
            #skimage.io.imsave(base_dir + new_raw_name, raw_img[0:tile_height,0:tile_width])
            #skimage.io.imsave(base_dir + new_labeled_name, labeled_img[0:tile_height,0:tile_width])
            #skimage.io.imsave(base_dir + new_boundary_name, boundary_img[0:tile_height,0:tile_width])
            f.write(new_raw_name+"\n"+new_labeled_name+"\n")
            #f.write(new_raw_name+"\n"+new_labeled_name+"\n"+new_boundary_name+"\n")
            count+=1
    f.close()

            