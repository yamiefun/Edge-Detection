import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# read input image
def get_input_image(image_name):
    # read color image as gray image
    img = cv2.imread(image_name, 0)
    return img


def get_argument():
    argp = argparse.ArgumentParser()
    argp.add_argument("-image",
                      help="input image name", required=True)
    argp.add_argument("-task",
                      help="choose assignment task", required=True,
                      type=int)
    argp.add_argument("-high",
                      help="high thresh ratio for double thresh",
                      type=float, default=0.09)
    argp.add_argument("-low",
                      help="low thresh ratio for double thresh",
                      type=float, default=0.05)
    args = argp.parse_args()
    return args


def LoG(image):
    kernal = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])
    kernal_size = 5
    width = (kernal_size-1)//2
    padded_img = np.pad(image, (width,), 'reflect')

    ret = np.zeros(padded_img.shape)
    for i in range(width, padded_img.shape[0]-width):
        for j in range(width, padded_img.shape[1]-width):
            ret[i][j] = (kernal*padded_img[i-width:i+width+1,
                         j-width:j+width+1]).sum()

    ret = ret[width:ret.shape[0]-width, width:ret.shape[1]-width]
    ret = np.clip(ret, 0, 255)
    ret = ret.astype("uint8")
    return ret


def gradient_filter(image, mode):
    kernal_x = None
    kernal_y = None
    if mode == "sobel":
        kernal_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernal_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif mode == "prewitt":
        kernal_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernal_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    kernal_size = 3
    width = (kernal_size-1)//2
    padded_img = np.pad(image, (width,), 'reflect')
    ret = np.zeros(padded_img.shape)
    grad_x = np.zeros(padded_img.shape)
    grad_y = np.zeros(padded_img.shape)

    for i in range(width, padded_img.shape[0]-width):
        for j in range(width, padded_img.shape[1]-width):
            grad_x[i][j] = (kernal_x*padded_img[i-width:i+width+1,
                            j-width:j+width+1]).sum()
            grad_y[i][j] = (kernal_y*padded_img[i-width:i+width+1,
                            j-width:j+width+1]).sum()
            # ret[i][j] = (grad_x[i][j]**2+grad_y[i][j]**2)**0.5
    grad_x = np.clip(grad_x, 0, 255)
    grad_y = np.clip(grad_y, 0, 255)
    ret = (grad_x**2+grad_y**2)**0.5
    ret = ret[width:ret.shape[0]-width, width:ret.shape[1]-width]
    # ret = ret/ret.max()*255
    ret = np.clip(ret, 0, 255)
    ret = ret.astype("uint8")
    return ret, grad_x, grad_y


# show both original image and processed image
def show_image(ori, image):
    disp = np.hstack((ori, image))
    cv2.imshow("image", disp)
    if cv2.waitKey():
        cv2.destroyAllWindows()


def save_image(image, filename):
    cv2.imwrite(filename, image)


def thresholding(image, thresh=60):
    ret = np.zeros(image.shape)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            ret[i][j] = 255 if image[i][j] >= thresh else 0
    return ret


def gaussian_smooth(image):
    kernal = np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]])
    kernal_size = 3
    width = (kernal_size-1)//2
    padded_img = np.pad(image, (width,), 'reflect')
    ret = np.zeros(padded_img.shape)
    for i in range(width, padded_img.shape[0]-width):
        for j in range(width, padded_img.shape[1]-width):
            ret[i][j] += (kernal*padded_img[i-width:i+width+1,
                          j-width:j+width+1]).sum()
    ret = ret[width:ret.shape[0]-width, width:ret.shape[1]-width]
    ret = np.clip(ret, 0, 255)
    ret = ret.astype("uint8")
    return ret


def non_max_sup(image, grad):
    height, width = image.shape
    ret = np.zeros(image.shape, dtype=np.int32)
    angle = grad * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, height-1):
        for j in range(1, width-1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = image[i, j+1]
                    r = image[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = image[i+1, j-1]
                    r = image[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = image[i+1, j]
                    r = image[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]

                if (image[i, j] >= q) and (image[i, j] >= r):
                    ret[i, j] = image[i, j]
                else:
                    ret[i, j] = 0
            except Exception:
                pass
    return ret


def double_thresholding(image, high_ratio, low_ratio):
    ret = np.zeros(image.shape, dtype=np.int32)
    strong = np.int32(255)
    weak = np.int32(25)
    high_thresh = image.max() * high_ratio
    low_thresh = image.max() * low_ratio
    strong_i, strong_j = np.where(image >= high_thresh)
    # zeros_i, zeros_j = np.where(image < low_thresh)
    weak_i, weak_j = np.where((image <= high_thresh) & (image >= low_thresh))
    ret[strong_i, strong_j] = strong
    ret[weak_i, weak_j] = weak
    return ret, strong, weak


def hysteresis(image, strong_value, weak_value):
    preserve_img = image.copy()
    while True:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] == weak_value:
                    try:
                        if ((image[i][j-1] == strong_value) or
                                (image[i][j+1] == strong_value) or
                                (image[i-1][j] == strong_value) or
                                (image[i+1][j] == strong_value) or
                                (image[i-1][j-1] == strong_value) or
                                (image[i-1][j+1] == strong_value) or
                                (image[i+1][j-1] == strong_value) or
                                (image[i+1][j+1] == strong_value)):
                            image[i][j] = strong_value
                    except Exception:
                        pass
        if not np.array_equal(image, preserve_img):
            preserve_img = image.copy()
        else:
            break

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == weak_value:
                image[i][j] = 0

    return image


def canny(image, high_ratio, low_ratio):
    ret = gaussian_smooth(image)
    ret, grad_x, grad_y = gradient_filter(ret, "sobel")
    grad = np.arctan2(grad_y, grad_x)
    ret = non_max_sup(ret, grad)
    ret, strong_value, weak_value = double_thresholding(ret,
                                                        high_ratio, low_ratio)
    ret = hysteresis(ret, strong_value, weak_value)

    return ret


def main():
    args = get_argument()
    try:
        os.mkdir("images/output")
    except Exception:
        pass
    if args.task == 1:
        img = get_input_image(args.image)
        # ret = gaussian_smooth(img)
        # mode = "LoG"
        # ret = LoG(img)
        mode = "sobel"
        ret, _, _ = gradient_filter(img, mode)
        for i in range(20, 150, 50):
            print(i)
            thr_img = thresholding(ret, thresh=i)
            # show_image(img, ret)
            img_name = os.path.split(args.image)
            img_name = img_name[-1]
            img_name = img_name[:-4]
            filename = f'images/output/{img_name}_{mode}_{i}.png'
            save_image(thr_img, filename)
            # save_image(ret, filename)

    elif args.task == 2:
        img = get_input_image(args.image)
        ret = canny(img, args.high, args.low)
        img_name = os.path.split(args.image)
        img_name = img_name[-1]
        img_name = img_name[:-4]
        filename = f'images/output/{img_name}_canny_{str(args.high)}_{str(args.low)}.png'
        save_image(ret, filename)


if __name__ == "__main__":
    main()
