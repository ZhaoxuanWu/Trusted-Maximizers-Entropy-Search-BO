import numpy as np


def used_idx():
    lefteyebrow = [49, 81, 68, 13, 17, 9, 7, 78]
    righteyebrow = [57, 64, 77, 76, 60, 11, 6, 10]
    lefteye = [80, 51, 79, 54, 45, 28, 14, 55]
    righteye = [58, 67, 40, 37, 41, 19, 20, 50]
    nose = [71, 53, 39, 36, 74, 66]
    mouth = [63, 1, 5, 52, 42, 43, 46, 3, 15, 70]
    face = [27, 26, 25, 23, 22, 21, 34, 2, 4, 16, 18, 32, 30, 31, 29]
    idx = lefteyebrow + righteyebrow + lefteye + righteye + nose + mouth + face

    return np.array(idx) - 1
