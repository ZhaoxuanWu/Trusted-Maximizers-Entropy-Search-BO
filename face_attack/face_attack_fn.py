import numpy as np
import cv2
import time
from PIL import Image
import face_recognition as fr
from os.path import join

from face_attack.utils.paths import *
from face_attack.synthesize.synthesize import synthesize
from face_attack.synthesize.synthesize_settings import FILE_EXTENSION
from face_attack.utils.common_methods import write_to_csv

TEST_IMAGE_ALIGNED_NPY_SANJAY_4_PATH = 'test_images/est_img_aligned_sanjay_4.npy'
TEST_IMAGE_ALIGNED_PTS_SANJAY_4_PATH = 'test_images/test_img_pts_sanjay_4.npy'
SAVE_IMAGE = False  # When true, every synthesized images are saved after getting generated

def fr_br_distance_func(alter_params, g_encodings, g_labels, debug=False, show_img=False):
    """
    Distance Function: calculates distance between the synthesized image and
    all the images in the gallery. Returns the 'distance' with the best match face.
    :param alter_params: parameters to synthesize/alter attacker's image
    :param g_encodings: Face features/encodings/embeddings of people in the gallery.
    :param g_labels: Label of the faces of people in the gallery.
    :param debug: If True, records (Images & CSV) will not be saved
    :param show_img: If True, each new (synthesized) image will be shown
    :return: Minimum distance
    """
    alter_params = np.array(alter_params, dtype=np.float32)
    alter_params = np.array(fix_eyeglass_parameter(alter_params), dtype=np.float32)
    syn_img, syn_img_name = synthesize(alter_params, write_to_csv_flag=False, save_img=False)

    # new_img = fr.load_image_file(syn_img)  # required if `synthesize` returns image path instead of image array
    new_img_encodings = fr.face_encodings(syn_img)[0]
    distances = fr.face_distance(g_encodings, new_img_encodings)
    min_dist = np.min(distances)
    best_match = g_labels[np.argmin(distances)]

    # print_run_summary(syn_img_name, min_dist, best_match, alter_params)  # remember: `print` costs runtime
    other_actions(syn_img_name, alter_params, best_match, min_dist, syn_img, is_debug=debug, is_save_img=SAVE_IMAGE,
                  is_show_img=show_img)

    return min_dist


def fix_eyeglass_parameter(alter_params):
    """
        Computes Euclidean distance between Eyeglass parameters in
        'alter_params' and MMDA model's parameters (EYEGLASS_PARAMS)
        Then, updates 'alter_params' to the nearest one in EYEGLASS_PARAMS.
        :param alter_params: numpy array of shape
        :return: updated_alter_params: numpy array
    """

    EG_START_IX = 5
    EG_END_IX = 6
    EG_PARAMS = np.array([[0.1443, -0.0833], [-0.1443, -0.0833], [0.0000, 0.1667]]) # Obtained from 'p0' of MMDA model v4
    EG_SCALE = 1  # Constant value for multiplying with eyeglass parameter values

    eg_params = alter_params[EG_START_IX:EG_END_IX + 1]
    distances = np.linalg.norm(EG_PARAMS - eg_params, axis=1)
    min_i = np.argmin(distances)

    updated_alter_params = alter_params
    updated_alter_params[EG_START_IX:EG_END_IX + 1] = EG_PARAMS[min_i] * EG_SCALE
    return updated_alter_params

def other_actions(syn_img_name, alter_params, best_match, min_dist, syn_img, is_debug, is_save_img, is_show_img):
    """
    Save record, Show image, or, Delete image after everything is done (Debug mode)
    :param syn_img:
    :param is_save_img:
    :param syn_img_name:
    :param alter_params:
    :param best_match:
    :param min_dist:
    :param is_debug:
    :param is_show_img:
    :return:
    """
    syn_img_full_path = ''

    if is_save_img:
        syn_img_full_path = join(SAVE_DIR_SYNTHESIZE, str(syn_img_name) + FILE_EXTENSION)
        print(syn_img_full_path)
        cv2.imwrite(syn_img_full_path, syn_img)

    if not is_debug:
        fields = [syn_img_name] + np.squeeze(alter_params).tolist() + \
                 [best_match] + [min_dist]
        write_to_csv(RESULTS_CSV_PATH, fields)

    ## SHOW IMAGE ##
    if is_show_img and is_save_img:
        img = Image.open(syn_img_full_path)
        img.show()

    if is_debug and is_save_img:
        os.remove(syn_img_full_path)
