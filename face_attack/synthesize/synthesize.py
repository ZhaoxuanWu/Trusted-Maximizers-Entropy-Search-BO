"""
Created by Sanjay at 7/8/2019

Feature: Enter feature name here
Enter feature description here
"""
import cv2
import time
import numpy as np
from PIL import Image

from os.path import join
from face_attack.synthesize.alter_image_utils import get_image_with_altered_attributes
from face_attack.synthesize.synthesize_settings import *
from face_attack.utils.common_methods import write_to_csv

NEW_ATTRIBUTES = NEW_ATTRIBUTES_len8
DEBUG_MODE = True  # When true, we don't save the synthesized image and write the record in CSV


def synthesize(new_attr, write_to_csv_flag=False, save_img=False, show_img=False):
    """
    Synthesize new face image with given "p" vector as `new_attr`
    :param new_attr: "p" vector
    :param write_to_csv_flag: If true, we record the synthesized image in CSV file
    :param save_img: If true we save the image in disk & return path, else just return the image
    :return: Path of the synthesized image or the image itself (depending on `save_img`)
    """
    # _time = int(time.time() - 1525000000)  # Get current timestamp
    _time = time.strftime("%Y%m%d-%H%M%S")

    new_attr = np.array(new_attr)
    new_attr = np.reshape(new_attr, (new_attr.shape[0], 1))  # Make it column vector
    new_attr = np.expand_dims(new_attr, axis=0)
    aligned_img, pts = np.load(TEST_IMAGE_ALIGNED_NPY_SANJAY_4_PATH), np.load(TEST_IMAGE_ALIGNED_PTS_SANJAY_4_PATH)
    out_img = get_image_with_altered_attributes(aligned_img, pts, new_attr)  # Get the altered image

    if not save_img:
        return out_img, _time

    # Save the new (synthesized) image
    file_name = join(SAVE_DIR_SYNTHESIZE, str(_time) + FILE_EXTENSION)
    cv2.imwrite(file_name, out_img)  # Save the image

    if show_img:
        img = Image.open(file_name)
        img.show()

    ## WRITE NEW PARAMETERS IN CSV ##
    if write_to_csv_flag:
        fields = [_time] + np.squeeze(new_attr).tolist()
        write_to_csv(SYNTHESIZE_RECORDS_PATH, fields)

    return file_name, _time


"""
Model: v4
======================
     1 (Beard)
======================
    0.0062   -0.1474    0.1411
    -0.1666    0.0779    0.0887
======================
     2 (Mark)
======================
    0.1025    0.0382   -0.2001    0.0594
    0.1438   -0.1826    0.0388    0.0000
    -0.1023   -0.0827   -0.0103    0.1953
======================
     3 (Eyeglass)
======================
   0.1443   -0.1443    0.0000
    -0.0833   -0.0833    0.1667
======================
     4 (Distortion)
======================
   -0.1179    0.1179
"""

if __name__ == '__main__':
    name = synthesize([0.5072, 0.0882, -0.3235, -0.142, -0.7086, 0.5772, -0.3332, 0.798], write_to_csv_flag=False, save_img=True, show_img=False)
    print(name)
