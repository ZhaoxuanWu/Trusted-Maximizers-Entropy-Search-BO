"""
Created by Sanjay at 7/24/2019

Feature: Enter feature name here
Enter feature description here
"""
from face_attack.utils.paths import *

##
## SETTINGS
##
FILE_EXTENSION = '.jpg'

# ================================================== #
#        SELECT MODEL and TEST IMAGE(PATH)           #
# ================================================== #
# mmda_model_path = MMDA_MODEL_PATH  # SANJAY - with all combinations
# mmda_model_path = MMDA_MODEL_V3_PATH  # SANJAY - without ROUNDED Eyeglasses
model_filename = 'mmda_sanjay_v4.mat'  # Change this to change the model
mmda_model_path = os.path.join(MODELS_DIR, model_filename)
img_path = TEST_IMAGE_SANJAY_4_PATH
# mmda_model_path = MMDA_MODEL_CAROLINE_PATH
# img_path = TEST_IMAGE_CAROLINE_PATH

# ================================================== #
#        NEW PARAMETERS FOR SYNTHESIZING             #
# ================================================== #
NEW_ATTRIBUTES_len7 = [-0.1, -0.2, 0.1, -0.1, 0.2, -0.09, 0.2]
NEW_ATTRIBUTES_len8 = [-0.1, -0.2, 0.1, -0.1, 0.2, -0.09, 0.2, 0.2]
