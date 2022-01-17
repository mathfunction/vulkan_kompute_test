import numpy as np
from cv2 import cv2


#calculating least square problem
def POS(xp,x):
    npts = xp.shape[1]

    A = np.zeros([2*npts,8])

    A[0:2*npts-1:2,0:3] = x.transpose()
    A[0:2*npts-1:2,3] = 1

    A[1:2*npts:2,4:7] = x.transpose()
    A[1:2*npts:2,7] = 1

    b = np.reshape(xp.transpose(),[2*npts,1])

    k,_,_,_ = np.linalg.lstsq(A,b, rcond=None)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx,sTy],axis = 0)

    return t,s

# resize and crop images
def resize_n_crop_img(img,t,s,target_size = 224):
    h0 = img.shape[0]
    w0 = img.shape[1]
    w = (w0/s*102).astype(np.int32)
    h = (h0/s*102).astype(np.int32)
    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)

    left = int(w/2 - target_size/2 + float((t[0] - w0/2)*102/s))
    right = left + target_size
    up = int(h/2 - target_size/2 + float((h0/2 - t[1])*102/s))
    below = up + target_size

    img = img[up:below, left:right]
    if img.shape[0] is not target_size or img.shape[1] is not target_size:
        return []
    img = np.expand_dims(img,0)

    return img


# resize and crop input images before sending to the R-Net
"""
img: face image (H, W, 3)
lm: face landmarks (5, 2) [[x0, y0], [x1, y1]...]
lm3D: landmarks for standard face (5, 3) 
           [[-0.31148657  0.29036078  0.13377953]
            [ 0.30979887  0.28972036  0.13179526]
            [ 0.0032535  -0.04617932  0.55244243]
            [-0.25216928 -0.38133916  0.22405732]
            [ 0.2484662  -0.38128236  0.22235769]]
"""
def align_img(img,lm,lm3D):

    h0 = img.shape[0]
    w0 = img.shape[1]

    # change from image plane coordinates to 3D sapce coordinates(X-Y plane)
    lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t,s = POS(lm.transpose(),lm3D.transpose())

    # processing the image
    img_new = resize_n_crop_img(img,t,s)
    trans_params = np.array([w0,h0,102.0/s,t[0],t[1]], dtype=object)

    return img_new, trans_params
    

if __name__ == "__main__":
    import argparse
    import os.path as osp
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--input', nargs='+')

    args = parser.parse_args()

    img = cv2.imread(args.input)
    save_txt = osp.splitext(args.input)[0] + '.txt'

    np.savetxt(save_txt, get_five_landmark(img), fmt='%.2f')