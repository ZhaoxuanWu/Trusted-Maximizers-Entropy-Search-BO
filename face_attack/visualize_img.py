import numpy as np
import face_recognition as fr
import time
import os
import cv2
import sys
sys.path.insert(1, '..')
from face_attack.utils.paths import *


def get_min_distance_and_best_match(p, g_encodings, g_labels):
    from face_attack.synthesize.synthesize import synthesize
    img, name = synthesize(p, save_img=False)
    new_img_encodings = fr.face_encodings(img)[0]
    distances = fr.face_distance(g_encodings, new_img_encodings)
    min_dist = np.min(distances)
    best_match = g_labels[np.argmin(distances)]
    return img, min_dist, best_match


def save_results(savedir, title, result, g_encodings_path, g_labels_path):

    g_encodings, g_labels = np.load(g_encodings_path), np.load(g_labels_path)

    _time = time.strftime("%Y%m%d-%H%M%S")

    final_img, final_min_dist, final_best_match = get_min_distance_and_best_match(result, g_encodings, g_labels)

    # if final_min_dist != result[1]:
    #     print('Mismatch in distance')
    #     # pass

    file_name = os.path.join(savedir, str(_time) + '.jpg')
    cv2.imwrite(file_name, final_img)  # Save the image

    f = open(os.path.join(savedir, str(_time) + '.txt'), "a")
    print('====================================================================== ', file=f)
    print('\t\t', title, file=f)
    print('====================================================================== ', file=f)
    print('Final Distance:', np.round(final_min_dist, 4), file=f)
    print('\t Final Best match:', final_best_match, file=f)
    print('', file=f)
    print('Final p vector:', np.array2string(np.round(result, 4), separator=', '), file=f)
    print('', file=f)
    f.close()


SAVE_DIR = os.path.join(PROJECT_HOME, 'results')
RESULTS_TITLE = 'Break in: ' + 'MMDA Model v4'
ENCODINGS_PATH = os.path.join(ATTACK_DIR, 'fr_gallery41_encodings.npy')
LABELS_PATH = os.path.join(ATTACK_DIR, 'fr_gallery41_me_encodings.npy')
# result = [-0.19321072, 0.16334914, 0.2, 0.03248319, 0.04204802, 0.09456136, 0.07896584, -0.2]
result = [-0.2, 0.1939494, 0.2, -0.09710134, 0.2, 0.09370947, -0.2, -0.2]

save_results(SAVE_DIR, RESULTS_TITLE, result, ENCODINGS_PATH, LABELS_PATH)