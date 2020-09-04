# Program To Read video
# and Extract Frames
import warnings
# Don't fear the future
warnings.simplefilter(action='ignore', category=FutureWarning)
# example of loading an image with the Keras API
# from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import cv2
# from PIL import Image
import numpy as np


def get_data_info(path, vid, scale_factor):
    vidObj = cv2.VideoCapture(path + vid)
    # get 3 of 4 video dimensions
    success, img = vidObj.read()
    height, width, depth = img.shape
    # Get the information of the incoming image type
    # dtmax = np.iinfo(img.dtype).max
    # count starts at one cuz we just read one
    count = 1
    # checks whether frames were extracted
    success = True
    while success:
        # vidObj object will extract frames
        success, img = vidObj.read()
        if not success:
            break
        # just get dimension
        count += 1
    # print(height)
    height = height // scale_factor
    width = width // scale_factor
    dims = (count, height, width, depth)
    return dims

def convert_to_vid(path, vid, sfactor=10, gray=True):
    dims = get_data_info(path, vid, sfactor)

    if gray:
        dims = dims[:-1] + (1,)

    vid_data = np.zeros(dims, dtype='float16')

    vidObj = cv2.VideoCapture(path + vid)
    success = True
    index = 0
    while success:
        success, img = vidObj.read()
        if not success:
            break
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (dims[2], dims[1]), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (dims[2], dims[1], dims[3]), interpolation=cv2.INTER_AREA)
        vid_data[index, :, :, 0] = img
        index += 1

    return vid_data


# Path to video file
path = "data/Video/"
vid = "1.mov"
folder = "frames2/"

vid = convert_to_vid(path, vid)

np.save("video1.npy", vid)

vid = convert_to_vid(path, "2.mov")

np.save("video2.npy", vid)
