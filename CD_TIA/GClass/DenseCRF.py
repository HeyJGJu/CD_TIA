import numpy as np
import pydensecrf.densecrf as dcrf

try:
    from cv2 import imread, imwrite
except ImportError:

    from skimage.io import imread, imsave
    imwrite = imsave

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


def CRFs(original_image_path,predicted_image_path,CRF_image_path):
    print("original_image_path: ",original_image_path)
    img = imread(original_image_path)

    #anno_rgb = imread(predicted_image_path).astype(np.uint32)
    anno_rgb = imread(predicted_image_path)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    colors, labels = np.unique(anno_lbl, return_inverse=True)
    

    HAS_UNK = 0 in colors
    if HAS_UNK:
         colors = colors[1:]
    

    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    #n_labels = len(set(labels.flat))
    n_labels = len(set(labels.flat)) - int(HAS_UNK)

    use_2d = False               
#
    #########################################################33
    if use_2d:                   

        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
    

        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)

        d.setUnaryEnergy(U)
    

        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
    

        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    else:

        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
    

        U = unary_from_labels(labels, n_labels, gt_prob=0.5, zero_unsure=None)  

        d.setUnaryEnergy(U)
    

        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=8,kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    

        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    


    Q = d.inference(10)
    

    MAP = np.argmax(Q, axis=0)
    

    MAP = colorize[MAP,:]
    imwrite(CRF_image_path, MAP.reshape(img.shape))
    print("CRF image:",CRF_image_path,"!")
if __name__ == "__main__":
    original_path='/disk/sdc/image'
    predicted_path='/disk/sdc/label'
    CRF_path='/disk/sdc/CRF'
    CRFs(original_path,predicted_path,CRF_path)