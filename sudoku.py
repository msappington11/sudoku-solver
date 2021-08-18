import time
import numpy as np
startTime = time.time()

def checkRow(row, grid, num):
    for i in range(9):
        if(grid[row][i] == num):
            return False
    return True

def checkCol(col, grid, num):
    for i in range(9):
        if(grid[i][col] == num):
            return False
    return True

def checkSquare(row, col, grid, num):
    startRow = (row//3) * 3 #uses integer division to get the start of the 3x3
    startCol = (col//3) * 3
    for i in range(startRow, startRow+3, 1):
        for j in range(startCol, startCol+3, 1):
            if(grid[i][j] == num):
                return False
    return True

def isValid(row, col, grid, num):
    return checkRow(row, grid, num) and checkCol(col, grid, num) and checkSquare(row, col, grid, num)

def solveSquare(row, col, grid, isDone):
    if((row == 8 and col == 8 and grid[row][col] != 0) or isDone[0]):
        isDone[0] = True
        return
    if(grid[row][col] == 0): # if the current position is empty
        for i in range(1, 10, 1): # loops through numbers 1-9 and tries to fit it into the empty square
            if ((row == 8 and col == 8 and grid[row][col] != 0) or isDone[0]):
                isDone[0] = True
                return
            if(isValid(row, col, grid, i)): # if the current number is valid
                grid[row][col] = i
                if(col < 8): # if the current position is not at the end of the row, calls itself on the square to the right
                    solveSquare(row, col+1, grid, isDone)
                elif(row < 8): # if the current position is at the end of the row, calls itself at the beginning of the next row
                    solveSquare(row+1, 0, grid, isDone)
        if(not isDone[0]):
            grid[row][col] = 0 # only do if its not done and keep track if its tracked all the way back mayvbe
    else: # if the current square is not blank, calls the next square
        if (col < 8):  # if the current position is not at the end of the row, calls itself on the square to the right
            solveSquare(row, col + 1, grid, isDone)
        elif (row < 8):  # if the current position is at the end of the row, calls itself at the beginning of the next row
            solveSquare(row + 1, 0, grid, isDone)

def solve(grid):
    isDone = [False]
    first_zero = np.argwhere(grid == 0)
    solveSquare(first_zero[0][0], first_zero[0][1], grid, isDone)
    return np.count_nonzero(grid) == 81

Sudoku = np.array([[0, 0, 0, 2, 7, 3, 9, 0, 5],
                   [5, 0, 0, 0, 0, 9, 0, 3, 7],
                   [7, 9, 0, 4, 0, 0, 0, 0, 2],
                   [0, 8, 0, 5, 2, 6, 4, 0, 0],
                   [1, 6, 5, 8, 0, 0, 0, 0, 0],
                   [0, 0, 2, 0, 9, 0, 5, 0, 6],
                   [0, 0, 1, 0, 0, 5, 3, 6, 0],
                   [9, 3, 8, 0, 6, 2, 0, 0, 0],
                   [0, 0, 0, 9, 3, 0, 0, 2, 8]])
solve(Sudoku)
print(time.time() - startTime)


