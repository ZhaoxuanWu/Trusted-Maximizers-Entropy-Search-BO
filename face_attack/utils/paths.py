import os

PROJECT_HOME_LINUX = r'./'
PROJECT_HOME = os.path.join(PROJECT_HOME_LINUX, 'face_attack')
DATASET_HOME = os.path.join(PROJECT_HOME, 'data')
MODELS_DIR = os.path.join(PROJECT_HOME, 'models')
ATTACK_DIR = os.path.join(DATASET_HOME, 'attacks')


# ============================================ #
#              REFERENCE and MASKS             #
# ============================================ #
REFERENCE_IMAGE_HOME = os.path.join(DATASET_HOME, 'reference_image')
REFERENCE_IMAGE_ALIGNED_NPY_PATH = os.path.join(REFERENCE_IMAGE_HOME, 'ref_img.npy')
REFERENCE_LANDMARKS_PATH = os.path.join(REFERENCE_IMAGE_HOME, 'reference_landmarks.npy')
MASK_FILLED_NPY_PATH = os.path.join(REFERENCE_IMAGE_HOME, 'ref_mask.npy')

# ============================================ #
#                   TEST IMAGES                #
# ============================================ #
TEST_IMAGES_DIR = os.path.join(DATASET_HOME, 'test_images')
TEST_IMAGE_SANJAY_4_PATH = os.path.join(TEST_IMAGES_DIR, 'test_image_sanjay_4.jpg')
TEST_IMAGE_ALIGNED_NPY_SANJAY_4_PATH = os.path.join(TEST_IMAGES_DIR, 'test_img_aligned_sanjay_4.npy')
TEST_IMAGE_ALIGNED_PTS_SANJAY_4_PATH = os.path.join(TEST_IMAGES_DIR, 'test_img_pts_sanjay_4.npy')

# ============================================ #
#       DIRECTORY TO SAVE GENERATED IMAGES     #
# ============================================ #
SAVE_DIR_SYNTHESIZE = os.path.join(DATASET_HOME, 'synthesize')
SYNTHESIZE_RECORDS_PATH = os.path.join(SAVE_DIR_SYNTHESIZE, 'synthesize_history.csv')

# ============================================ #
#               RESULTS directory              #
#       Raw outputs are saved as TXT files     #
# ============================================ #
RESULTS_CSV_PATH = os.path.join(PROJECT_HOME, 'results', 'history.csv')