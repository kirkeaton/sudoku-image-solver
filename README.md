Sudoku Image Solver
===================

### Overview

This is a program that attempts to solve Sudoku puzzles that have been extacted from an image.

### How it works

Through various image processing techniques combined with optical character recognition (OCR) this program
will try to extract the digits from a Sudoku puzzle. Once the digits are extracted, it will attempt
to solve it. The process works as follows:

1. Find the main grid for the puzzle
2. Find the four corners associated with this grid
3. Perform a homology to extract just the puzzle from the image
4. Perform a form of thresholding if necessary
5. Extract each cell from the puzzle
6. Perform OCR to determine the digits
7. Solve the puzzle

### What is included

1. Test data to train a support vector machine used for optical character recognition
2. Various test puzzles with the ground truth values used for comparison.

### Required libraries

1. OpenCV
2. OpenCV
3. Numpy
4. PyLab
5. LibSVM
6. Python Imaging Library (PIL)
7. Scipy

### Execution

To run the program:

type `python imsudoku.py`

### Results

The average accuracy of extracting the initial digits was 86%.

![Sudoku extraction](http://i.imgur.com/GK1rx5s.png)
