import numpy as np
import os
from scipy.io import loadmat
from scipy import interpolate
from numba import jit
""" python codes for matlab applycform(rgb_img, makecform('xyz2srgb') )
	colorspace detail:
		srgb : IEC 61966-2-1, use ITU-R BT.709 primaries and a transfer function (gamma curve) typical of CRTs.
		xyz: CIE 1931 xyz
"""

linearBGR_fwd = np.array([ [0.1431, 0.0606, 0.7141],
    					[0.3851, 0.7169, 0.0971],
    					[0.4361, 0.2225, 0.0139]
					])
linearBGR_bwd = np.linalg.inv(linearBGR_fwd)
# adapBGR = np.array([[-5.3243e-5, 1.6591e-5, 1		 ],
#     				[-1.7476e-5, 1.0000, 	1.9485e-5],
#    					[0.9997,	 1.8011e-5, -1.069e-5]
#    					])
adapXYZ = np.array([[ 1,          -1.6591e-5, 5.3256e-5],
    				[ -1.9485e-5, 1.0000,    1.748e-5],
   					[ 1.0693e-5,  -1.8016e-5, 1.0003]
   					], dtype=np.float32)

color_dir = os.path.dirname(os.path.abspath(__file__))
def bgr_revliner(img):
	""" convert linear bgr to bgr
	"""
	out_img = np.copy(img)
	out_img = out_img.dot(linearBGR_bwd)

	out_img[out_img>1] = 1
	out_img[out_img<0] = 0

	# handel curve
	TRC = loadmat(os.path.join(color_dir,'TRC_bwd.mat'))['TRC']
	TRC = np.squeeze(TRC) 

	for channel in range(3):
		lut1d = TRC[2-channel].squeeze()
		samples = np.linspace(0, 1.0, len(lut1d))
		# monotonicize
		xi, yi = monotonicize(lut1d, samples)
		out_img[:,channel] = interpolate.spline(xi, yi, out_img[:,channel])

	# out_img value range
	out_img[out_img<0] = 0
	out_img[out_img>1] = 1
	return out_img

@jit(nopython=True)
def monotonicize(xin, yin):
	xout = []
	yout = []

	iin = 0
	while iin<len(xin)-1 and xin[iin+1]<= xin[iin]:
		iin = iin+1

	igood = iin
	iout =0
	xout.append(xin[igood])
	yout.append(yin[igood])

	while iin <len(xin):
		if xin[iin]>xin[igood]:
			igood = iin
			iout = iout +1
			xout.append(xin[igood])
			yout.append(yin[igood])
		iin = iin+1

	xout = np.array(xout)
	yout = np.array(yout)
	return xout, yout

@jit(nopython=True)
def xyzadapt(img):
	# adaption for linear BGR
	img = img.dot(adapXYZ.T)
	return img

def XYZ2sBGR(img):
	# convert bgr image to xyz image, according to matlab color conversion method	

	out_img = np.copy(img)
	out_img = np.float32(out_img)
	
	out_img = out_img.reshape((out_img.shape[0]*out_img.shape[1], out_img.shape[2]))

	out_img = xyzadapt(out_img) 
	out_img = bgr_revliner(out_img)
	out_img = out_img.reshape(img.shape)
	return out_img