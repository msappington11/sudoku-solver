import cv2
from PIL import Image
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from sudoku import solve

def filterContours(contours):
    if(1600 < cv2.contourArea(contours) < 3500):
        return True
    return False

def trimEdges(cornerGrid):
    for corner in cornerGrid:
        corner[0][0] += 2
        corner[0][1] += 2
        corner[1][0] -= 2
        corner[1][1] -= 2
    return cornerGrid

def merge(left, right):
    # return the opposite side if the other side is empty
    if(len(left) == 0):
        return right
    if(len(right) == 0):
        return left

    result = []
    leftIndex = rightIndex = 0
    while(len(result) < len(left) + len(right)): # loop until the result has all the elements from the right and left
        if(left[leftIndex][0][0] <= right[rightIndex][0][0]): # if the left X coord is less than the right X coord
            result.append(left[leftIndex])
            leftIndex+=1
        else:
            result.append(right[rightIndex])
            rightIndex+=1
        if(rightIndex == len(right)): # if the right side reaches the end, add the rest of the left side
            result += left[leftIndex:]
            break
        if(leftIndex == len(left)):
            result += right[rightIndex:]
            break
    return result

def sort(arr):
    if(len(arr) < 2): # return if the array is 1 element
        return arr
    midpoint = len(arr)//2
    return merge(sort(arr[:midpoint]), sort(arr[midpoint:]))

def removeShadow(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result


def scan():
    cam = cv2.VideoCapture(1)
    while True:
        #img = Image.open('paperCropped.jpg') # read image
        # img = np.asarray(img) # convert to array
        cap, img = cam.read() # read camera input
        img = img[0:504, 0:504] # get only the top left 500x500 pixels
        img = removeShadow(img) # removes the shadow in the image
        #img = cv2.resize(img, (504, 504)) # resize

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to greyscale
        grey = cv2.GaussianBlur(grey, (5, 5), 1) # blur
        ret, thresh = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY) # create threshold image
        contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find the contours around the white spaces from the threshold
        contours = list(filter(filterContours, contours)) # filter out the contours that are too big or small to be squares
        if(len(contours) == 81):
            cam.release()
            cv2.destroyAllWindows()
            return contours, thresh

        cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
        cv2.imshow('scan', img)
        cv2.waitKey(1)

def make2D(oneD):
    twoD = []
    for i in reversed(range(9)):
        row = []
        for j in reversed(range(9)):
            row.append(oneD[i*9+j])
        twoD.append(row)
    return twoD

def findCorners(contours): # gets the corners of each contour (each box of the puzzle)
    corners = []
    for contour in contours:
        minX = contour[0][0][0]
        minY = contour[0][0][1]
        maxX = contour[0][0][0]
        maxY = contour[0][0][1]
        for i in range(len(contour)):
            if (contour[i][0][0] < minX):
                minX = contour[i][0][0]
            elif (contour[i][0][0] > maxX):
                maxX = contour[i][0][0]
            if (contour[i][0][1] < minY):
                minY = contour[i][0][1]
            elif (contour[i][0][1] > maxY):
                maxY = contour[i][0][1]
        corners.append([[minX, minY], [maxX, maxY]])  # top left corner and bottom right corner

    corners = make2D(corners) # converts the 1D list of corners into a 2D array of corners that more accurately displays a sudoku puzzle
    for i in range(9):
        corners[i] = sort(corners[i]) # sorts the squares in each row
    return(corners)

def getSudokuGrid(cornerGrid, threshold):
    model = tf.keras.models.load_model('model')  # loads tensorflow model to predict numbers
    trimmed = list(map(trimEdges, cornerGrid))  # trims off 2 pixels from each side so none of the lines get read as numbers
    sudokuGrid = np.zeros((9, 9), int)
    for i in range(9):
        for j in range(9):
            number = 0
            square = threshold[trimmed[i][j][0][1]:trimmed[i][j][1][1],
                     trimmed[i][j][0][0]:trimmed[i][j][1][0]]  # gets the image of the square from the corners
            cornerGrid[i][j] = square  # replaces the corners with the image of the square
            shape = np.shape(square)
            blackPixels = shape[0] * shape[1] - cv2.countNonZero(square)
            if (blackPixels > 120):  # if the square has more than 200 black pixels indicating there is a letter
                # prepares the image to be fed into the network
                square = cv2.resize(square, (28, 28))
                square = abs(1 - (square / 255.0))
                square = np.array(square, dtype=np.uint8)
                # predicts the number
                prediction = model.predict(square[np.newaxis, ..., np.newaxis])
                classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                number = classes[np.argmax(prediction)]
            sudokuGrid[i][j] = number
    return sudokuGrid

def test():
    contours, threshold = scan() # scans for squares of a sudoku puzzle, once the squares are found, returns the contours of the squares and a threshold image of the puzzle
    cornerGrid = findCorners(contours) # finds the corners of the contours and returns a 2d array of of the corners in a 9x9 sudoku grid
    grid = getSudokuGrid(cornerGrid, threshold) # convers the 9x9 grid of corners into a sudoku grid represented as a numpy array

    print(grid)
    isSolved = solve(grid)
    print(grid)
