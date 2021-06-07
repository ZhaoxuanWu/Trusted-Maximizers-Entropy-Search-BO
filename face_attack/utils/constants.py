"""
Created by Sanjay at 5/28/2019

Feature: Enter feature name here
Enter feature description here
"""

"""
Important values
"""
CHIN_OFFSET_IN_MASK_X = 0  # pixels (around the chin)
CHIN_OFFSET_IN_MASK_Y = 10  # pixels (around the chin)

"""
Labels for the training images
"""
LABELS = {}
LABELS['clean'] = 1  # NO_BEARD
LABELS['bearded'] = 2  # BEARD
LABELS['moustache'] = 3  # MOUSTACHE

LABELS['noScar'] = 1  # NO_SCAR
LABELS['scarCheek'] = 2  # SCAR_CHEEK
LABELS['scarForehead'] = 3  # SCAR_FOREHEAD
LABELS['mole'] = 4  # SCAR_MOLE

LABELS['noEyeglass'] = 1  # NO_EYEGLASS
LABELS['square'] = 2  # EYEGLASS_SQUARE
LABELS['round'] = 3  # EYEGLASS_ROUND

LABELS['noExpression'] = 1  # NO_EXPRESSION
LABELS['distortion'] = 2  # EXPRESSION_DISTORTION
