import numpy as np
from PIL import Image


def crop_image(image_array, point, size):
    """
    Cropping the image into the assigned size

    image_array: numpy array of image
    size: desirable cropped size

    return -> cropped image array
    """
    img_height, img_width = point  # assigned location in crop
    # for color image
    if len(image_array.shape) == 3:
        image_array = image_array[:, img_height:img_height + size[0], img_width:img_width + size[1]]
    # for gray image
    elif len(image_array.shape) == 2:
        image_array = image_array[img_height:img_height + size[0], img_width:img_width + size[1]]
    return image_array


def crop_center_point(image_array, size):
    """

    Finding point for center crop for gray scale

    image_array: numpy array of image
    size: desirable width & height as input
$
    return -> cropped the center of input image into given size
    """
    img_height, img_width = image_array.shape
    crop_height, crop_width = size
    location_crop = ((img_height - crop_height)//2, (img_width-crop_width)//2)
    return location_crop


def rotate_image(image_array, degree):
    """
    Rotating the image array by assigned degree

    image_array: numpy array of image
    degree: rotating as counterclockwise by 90, 180, 270, 0
            degree = number of times the array rotated; should be 0,1,2,3...

    return -> rotated image array
    """
    # for color image
    if len(image_array.shape) == 3:
        image_array = np.rot90(image_array, degree, axes=(1, 2))
    elif len(image_array.shape) == 2:
        image_array = np.rot90(image_array, degree)
    return image_array


def flip_image(image_array, axis=1):
    """
    Flipping the image upside down or left/right

    image_array: numpy array of image
    axis: 0 or 1 (flipping upside down or flipping left/right )

    return -> flipped image array
    """
    # axis = 0 : flipping upside down , axis = 1 : flipping left/right
    if len(image_array.shape) == 3:
        image_array = np.transpose(image_array, (1, 2, 0))
        image_array = np.flip(image_array, axis=axis)
        image_array = np.transpose(image_array, (2, 0, 1))
    elif len(image_array.shape) == 2:
        image_array = np.flip(image_array, axis)
    return image_array


def change_brightness(image_array, brightness_degree):
    """
    Changing the brightness of the image

    image_array: numpy array of image
    brightness_degree: the degree of wanted brightness

    return -> brightness changed image array
    """
    # print(image_array.shape)
    if len(image_array.shape) == 3:
        d, h, w = image_array.shape[0], image_array.shape[1], image_array.shape[2]
        value = [[[brightness_degree]*w]*h]*d
    elif len(image_array.shape) == 2:
        h, w = image_array.shape[0], image_array.shape[1]
        value = [[brightness_degree]*w]*h
    value = np.array(value)
    image_array = image_array + value
    image_array[image_array > 255] = 255
    # print('inside', image_array.dtype)    ###-----------------------------------------------------
    return image_array


def uniform_noise_image(image_array, low, high):
    """
    Adding the uniform noise to the image

    image_array : numpy array of image
    low, high : the low and high boundary of noise value

    return -> uniform noise added image array
    """
    # shape; the size of the image array
    shape = image_array.shape
    image_array = image_array + np.random.uniform(low, high, shape)
    image_array[image_array > 255] = 255
    # changing back from float64 to uint8
    image_array = image_array.astype(np.uint8)
    return image_array


def array_to_image(image_array):
    if len(image_array.shape) == 3:
        image_array = np.transpose(image_array, (1, 2, 0))
    back_to_img = Image.fromarray(image_array)
    return back_to_img


if __name__ == "__main__":
    # Read image
    img = Image.open('../data/train_data/vid01_041_1.png')
    img_array = np.array(img)
    # transpose back to (h, w, d)
    if img_array.shape == 3:
        img_array = np.transpose(img_array, (2, 0, 1))
    # array_to_image(img_array).show()

    #test for cropping image
    cropped_image = crop_image(img_array, (3, 5), (224, 224))
    array_to_image(cropped_image).show()

    # test for center crop
    location_to_crop = crop_center_point(img_array, (224, 224))
    img_array = crop_image(img_array,location_to_crop, (224, 224))
    array_to_image(img_array).show()

    # test for rotating image
    rotated_image = rotate_image(cropped_image, 1)
    array_to_image(rotated_image).show()

    # test for flipping image
    flipped_image = flip_image(rotated_image, 1)
    array_to_image(flipped_image).show()

    # test for changing brightness of image
    brightness_changed_image = change_brightness(flipped_image, 30)
    brightness_changed_image = np.uint8(brightness_changed_image)   ### why the type changed??!?!?!?
    array_to_image(brightness_changed_image).show()

    # test for uniform noise image
    uniform_noised_image = uniform_noise_image(brightness_changed_image, 0, 50)
    array_to_image(uniform_noised_image).show()
