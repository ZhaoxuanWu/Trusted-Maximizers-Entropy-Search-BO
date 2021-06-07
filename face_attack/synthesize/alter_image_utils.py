import cv2
import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_erosion
from face_attack.preprocessing.preprocessing_utils.UsedIDX import used_idx
from face_attack.preprocessing.preprocessing_utils.colorspace import sBGR2XYZ, XYZ2BGR
from face_attack.preprocessing.preprocessing_utils.warp import warp_image
from face_attack.synthesize.synthesize_settings import *

# ref_img = cv2.imread(REFERENCE_IMAGE_ALIGNED_PATH, 1)
ref_img = np.load(REFERENCE_IMAGE_ALIGNED_NPY_PATH)
w, h, d = ref_img.shape
idx = used_idx()
ref_coords = np.load(REFERENCE_LANDMARKS_PATH)[idx, :]
# ref_mask = cv2.imread(MASK_FILLED_PATH, 1)
ref_mask = np.load(MASK_FILLED_NPY_PATH)


def get_image_with_altered_attributes(img, used_pts, new_attributes):
    used_pts = used_pts[idx, :]
    wrapped_img = warp_image(img, used_pts, ref_coords, ref_img.shape)
    wrapped_img = cv2.bitwise_and(wrapped_img, ref_mask)

    # id = random.randint(1,5)
    M = loadmat(mmda_model_path)['mmda_model']

    # scale BGR then convert BGR to XYZ based matlab method
    SCALE = max(w, h)
    im = sBGR2XYZ.sBGR2XYZ(wrapped_img)

    coords = used_pts / float(SCALE)

    test_faces = im[:, :, 0].T.flatten()
    test_faces = np.concatenate((test_faces, im[:, :, 1].T.flatten()))
    test_faces = np.concatenate((test_faces, im[:, :, 2].T.flatten()))
    test_faces = np.concatenate((test_faces, coords.T.flatten()))
    test_faces = test_faces.reshape((len(test_faces), 1))

    outputs = alter_attr(test_faces, M, new_attributes)

    # reshape
    outputs = outputs.squeeze()
    new_img = np.zeros((w, h, d))
    i = 0
    for channel in range(d):
        new_img[:, :, channel] = np.reshape(outputs[i:i + w * h], (h, w)).T
        i = i + w * h

    # xyz to bgr
    out_img = XYZ2BGR.XYZ2sBGR(new_img)
    out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min()) * 255
    out_img = np.uint8(out_img)
    out_img = cv2.bitwise_and(out_img, ref_mask)
    # cv2.imwrite('out_img.png', out_img)

    out_pts = outputs[w * h * d:].reshape(2, len(ref_coords)).T
    out_pts = out_pts * SCALE

    # un-warp
    out_img = warp_image(out_img, ref_coords, out_pts, ref_img.shape)
    mask_img = warp_image(ref_mask, ref_coords, out_pts, ref_img.shape)
    # cv2.imwrite('testDeWarp.png', out_img)
    # cv2.imwrite('testMaskDeWarp.png', mask_img)
    result = postp(out_img, mask_img, img)
    return result


def alter_attr(im, mmda, new_attributes):
    if len(im.shape) == 1:
        im = im.reshape(im.shape[0], 1)

    r, c = im.shape
    Q = mmda['Q'][0, 0]
    P = mmda['P'][0, 0]
    Pr = mmda['Pr'][0, 0]
    C = mmda['C'][0, 0].squeeze()

    org = mmda['org'][0, 0]
    lowObs = mmda['lowObs'][0, 0]

    coefficients = np.dot(Q.T, np.dot(P.T, im - np.tile(org, (1, c))))  # p
    attr_end = int(C.sum() - len(C))
    old_attributes = coefficients[:attr_end, :]

    # new_attr = np.append(new_gender*intensity[0], new_race*intensity[1])
    # new_attr = np.append(new_attr, new_age*intensity[2])
    # new_attr = new_attr.reshape((len(new_attr),1))
    # print ('old_attr', old_attr)
    # print ('new_attr', new_attr)

    # mask
    coefficients[:attr_end, :] = new_attributes
    # coeffs[:attr_end,:] = new_attributes * attr_mask + old_attr*(1-attr_mask)
    out_img = np.dot(Pr, np.dot(Q, coefficients)) + np.tile(org, (1, c))
    return out_img


def postp(out_img, mask, img):
    if len(mask.shape) < 3:
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

    for i in range(mask.shape[2]):
        mask[:, :, i] = grey_erosion(mask[:, :, i], size=(7, 7))

    mask = mask.squeeze()
    mask = gaussian_filter(mask, sigma=2.5)
    mask = np.float32(mask)
    mask = mask / mask.max()
    img1 = np.float32(out_img)
    img2 = np.float32(img)
    img = img1 * mask + img2 * (1 - mask)
    img = np.uint8(img)

    return img

if __name__ == '__main__':
    pass