import cv2
import homography
import numpy as np
import os
from libsvm.svmutil import *
from PIL import Image
from pylab import *
from scipy import ndimage

H = 1

# helper functions for the sudoku solver
#   Ref(s):
#   http://stackoverflow.com/questions/201461/shortest-sudoku-solver-in-python-how-does-it-work
def same_row(i, j):
    return i / 9 == j / 9


def same_col(i, j):
    return (i - j) % 9 == 0


def same_block(i, j):
    return i / 27 == j / 27 and i % 9 / 3 == j % 9 / 3


# function that solves a sudoku puzzle
#   Ref(s):
#   http://stackoverflow.com/questions/201461/shortest-sudoku-solver-in-python-how-does-it-work
def solve_puzzle(a):
    i = a.find("0")
    if i == -1:
        print("solved")
        # puzzle is solved, format the output
        soln = []
        for j in range(81):
            soln.append(int(a[j]))
        print((array(soln).reshape(9, 9)))
        return None

    # determine any excluded numbers
    excluded_numbers = set()
    for j in range(81):
        if same_row(i, j) or same_col(i, j) or same_block(i, j):
            excluded_numbers.add(a[j])

    for m in "123456789":
        if m not in excluded_numbers:
            # At this point, m is not excluded by any row, column, or block, so let's place it and recurse
            return solve_puzzle(a[:i] + m + a[i + 1 :])


# function that performs a homography based on four points
#   Ref(s):
#   Solem, J.E., Programming Computer Vision with Python, O'Reilly (2012)
def perform_homography(x):
    global H
    fp = array([array([p[1], p[0], 1]) for p in x]).T
    tp = array([[0, 0, 1], [0, 1000, 1], [1000, 1000, 1], [1000, 0, 1]]).T
    # estimate the homography
    H = homography.H_from_points(tp, fp)


# helper function for geometric_transform
#   Ref(s):
#   Solem, J.E., Programming Computer Vision with Python, O'Reilly (2012)
def warpfcn(x):
    x = array([x[0], x[1], 1])
    xt = dot(H, x)
    xt = xt / xt[2]
    return xt[0], xt[1]


# finds the edges of a straigtened sudoku puzzle
def find_sudoku_edges(im, axis=0):
    size = im.shape[axis]
    x = []
    for i in range(10):
        x.append((i * size) / 9)

    return x


# resizes an image
def imresize(im, sz):
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))


# computes a feature vector for an ocr image patch
#   Ref(s):
#   Solem, J.E., Programming Computer Vision with Python, O'Reilly (2012)
def compute_feature(im):
    # resize and remove border
    norm_im = imresize(im, (30, 30))
    norm_im = norm_im[3:-3, 3:-3]
    return norm_im.flatten()


# returns labels & ocr features for all images in the path
#   Ref(s):
#   Solem, J.E., Programming Computer Vision with Python, O'Reilly (2012)
def load_ocr_data(path):
    # get a list of all the images
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    # create labels
    labels = [int(imfile.split("/")[-1][0]) for imfile in imlist]

    # create features from the images
    features = []
    for imname in imlist:
        im = array(Image.open(imname).convert("L"))
        features.append(compute_feature(im))
    return array(features), labels


def main():
    print("Generating OCR data")
    print("===================")
    # training data
    features, labels = load_ocr_data("ocr_data/training/")

    # testing data
    test_features, test_labels = load_ocr_data("ocr_data/testing/")

    # train a linear SVM classifier
    features = list(map(list, features))
    test_features = list(map(list, test_features))

    prob = svm_problem(labels, features)
    param = svm_parameter("-t 0 -q")

    m = svm_train(prob, param)

    # how did the training do?
    res = svm_predict(labels, features, m)

    # how does it perform on the test set?
    res = svm_predict(test_labels, test_features, m)

    print("")

    # process all 60 of our images
    for fp in range(1, 61):
        # input & actual results
        infile = "sudoku_images/sudoku%d.jpg" % (fp)
        ground = "sudoku_images/sudoku%d.sud" % (fp)

        # Ref(s) for lines 106 to 131
        #   http://stackoverflow.com/a/11366549

        # read the image, blur it
        # create a structuring element to pass to
        orig = cv2.imread(infile, 0)
        blur = cv2.GaussianBlur(orig, (11, 11), 0)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        # perform a morphology based on the previously computed kernel
        close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel1)
        div = np.float32(blur) / (close)
        res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
        res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

        # perform an adaptive threshold and find the contours
        thresh = cv2.adaptiveThreshold(res, 255, 0, 1, 19, 2)
        contours, hier = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # find the sudoku gameboard by looking for the largest square in image
        biggest = None
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 100:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area

        # calculate the center of the square
        M = cv2.moments(biggest)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # find the location of the four corners
        for a in range(0, 4):
            # calculate the difference between the center
            # of the square and the current point
            dx = biggest[a][0][0] - cx
            dy = biggest[a][0][1] - cy

            if dx < 0 and dy < 0:
                topleft = (biggest[a][0][0], biggest[a][0][1])
            elif dx > 0 and dy < 0:
                topright = (biggest[a][0][0], biggest[a][0][1])
            elif dx > 0 and dy > 0:
                botright = (biggest[a][0][0], biggest[a][0][1])
            elif dx < 0 and dy > 0:
                botleft = (biggest[a][0][0], biggest[a][0][1])

        # the four corners from top left going clockwise
        corners = []
        corners.append(topleft)
        corners.append(topright)
        corners.append(botright)
        corners.append(botleft)

        # perform the homography
        perform_homography(corners)
        # perform a geometric transform to get just the puzzle in our image
        fixed = array(
            Image.fromarray(
                ndimage.geometric_transform(blur, warpfcn, (1000, 1000)), "L"
            )
        )

        # perform adaptive thresholding increase the contrast between paper and ink
        fixed = cv2.adaptiveThreshold(
            fixed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # find the x and y edges
        x = find_sudoku_edges(fixed, axis=0)
        y = find_sudoku_edges(fixed, axis=1)

        # crop each cell and add it to a list of crops
        crops = []
        for col in range(9):
            for row in range(9):
                crop = fixed[
                    int(y[col]) : int(y[col + 1]), int(x[row]) : int(x[row + 1])
                ]
                crops.append(compute_feature(crop))

        print(("Puzzle #%02d" % (fp)))
        print("==========")
        # check our results and formulate it into a 9x9 array
        res, acc, vals = svm_predict(
            loadtxt(ground).reshape(81), list(map(list, crops)), m
        )
        res_im = array(res).reshape(9, 9)

        print("Result:")
        print(res_im)
        print("")

        if acc[0] >= 100:
            print("Puzzle extracted perfectly. Solving now.")
            print("")
            puzz = ""
            puzzle = res_im.flatten()
            for i in range(len(puzzle)):
                puzz += str(int(puzzle[i]))
            print("Solution:")
            solve_puzzle(puzz)
            print("")
        else:
            print("Failed to extract puzzle perfectly. Cannot solve.")
            print("")


if __name__ == "__main__":
    main()
