import numpy as np
import os
from scipy.io import loadmat
from scipy import interpolate
from numba import jit

""" python codes for matlab applycform(rgb_img, cform('srgb2xyz') )
	colorspace detail:
		srgb : IEC 61966-2-1, use ITU-R BT.709 primaries and a transfer function (gamma curve) typical of CRTs.
		xyz: CIE 1931 xyz
"""

linearBGR_fwd = np.array([[0.1431, 0.0606, 0.7141], [0.3851, 0.7169, 0.0971], [0.4361, 0.2225, 0.0139]])
# adapBGR = np.array([[-5.3243e-5, 1.6591e-5, 1		 ],
#     				[-1.7476e-5, 1.0000, 	1.9485e-5],
#    					[0.9997,	 1.8011e-5, -1.069e-5]
#    					])
adapBGR = np.array([[1, 1.6591e-5, -5.3243e-5, ], [1.9485e-5, 1.0000, -1.7476e-5], [-1.069e-5, 1.8011e-5, 0.9997]])

color_dir = os.path.dirname(os.path.abspath(__file__))


#
#  convert bgr to linear bgr
#
def bgrlinear(img):
    out_img = np.copy(img)
    # handel curve

    TRC = loadmat(os.path.join(color_dir, 'TRC.mat'))['TRC']
    TRC = np.squeeze(TRC)

    for channel in range(3):
        lut1d = TRC[2 - channel].squeeze()
        samples = np.linspace(0, 1.0, len(lut1d))
        out_img[:, channel] = interpolate.spline(samples, lut1d, out_img[:, channel])

    # out_img value range
    out_img[out_img < 0] = 0
    out_img[out_img > 1] = 1
    out_img = out_img.dot(linearBGR_fwd)
    return out_img

@jit(nopython=True)
def bgradapt(img):
    # adaption for linear BGR
    img = img.dot(adapBGR.T)
    return img


def sBGR2XYZ(img):
    # convert bgr image to xyz image, according to matlab color conversion method

    out_img = np.copy(img)
    out_img = np.float32(out_img)
    out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())

    out_img = out_img.reshape((out_img.shape[0] * out_img.shape[1], out_img.shape[2]))

    out_img = bgrlinear(out_img)
    out_img = bgradapt(out_img)
    out_img = out_img.reshape(img.shape)
    return out_img
