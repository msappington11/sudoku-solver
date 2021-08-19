from tkinter import filedialog
import tkinter as tk
import cv2
import helpers
from PIL import Image, ImageTk
from threading import Thread
import numpy as np


class MainFrame:
    def __init__(self, master):
        # sets up frame and background
        self.master = master
        self.frame = tk.Frame(width=700, height=700, master=self.master)
        background_image = tk.PhotoImage(file='assets/main_background.png')
        self.background = tk.Label(image=background_image, master=self.frame)
        self.background.photo = background_image
        self.background.place(x=0, y=0)
        self.frame.pack()
        self.frame.pack_propagate(0)

        #scan button
        scan_image = tk.PhotoImage(file='assets/scan.png')
        self.scanButton = tk.Button(image=scan_image, master=self.frame, command=self.openScanFrame, highlightthickness = 0, bd = 0)
        self.scanButton.photo = scan_image
        self.scanButton.place(x=230, y=200)

        # from images button
        from_images = tk.PhotoImage(file='assets/from_images.png')
        self.photoButton = tk.Button(image=from_images, master=self.frame, command=self.openFile, highlightthickness = 0, bd = 0)
        self.photoButton.photo = from_images
        self.photoButton.place(x=220, y=300)

        # manual entry button
        manual_image = tk.PhotoImage(file='assets/manual.png')
        self.manualButton = tk.Button(image=manual_image, master=self.frame, command=self.openSudoku, highlightthickness = 0, bd = 0)
        self.manualButton.photo = manual_image
        self.manualButton.place(x=250, y=400)

    def openScanFrame(self):
        ScanFrame(self.master)
        self.frame.pack_forget()

    def openFile(self):
        path = filedialog.askopenfilename(filetypes=[('Image files', '.png .jpg .jpeg')])
        if(path == ''):
            return
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        contours, threshold = self.getGrid(img)
        if(contours == -1):
            self.error = tk.Label(text='Error', master=self.frame)
            self.error.place(x=350, y=350)
        else:
            corners = helpers.findCorners(contours)
            grid = helpers.getSudokuGrid(corners, threshold)
            SudokuDisplay(self.master, grid)
            self.frame.pack_forget()

    def getGrid(self, img): # gets the grid from the image
        img = cv2.resize(img, (500, 500))  # resize
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to greyscale
        grey = cv2.GaussianBlur(grey, (5, 5), 1)  # blur
        ret, thresh = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY)  # create threshold image
        contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # find the contours around the white spaces from the threshold
        contours = list(filter(helpers.filterContours, contours))  # filter out the contours that are too big or small to be squares
        cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
        if (len(contours) == 81):
            return contours, thresh
        return -1, -1

    def openSudoku(self):
        SudokuDisplay(self.master)
        self.frame.pack_forget()


class ScanFrame():
    def __init__(self, master):
        # variables needed for functions
        self.isScanning = False
        self.grid = [[None]]

        # sets up frame and background
        self.master = master
        self.frame = tk.Frame(width=700, height=700, master=self.master)
        background_image = tk.PhotoImage(file='assets/scan_background.png')
        self.background = tk.Label(image=background_image, master=self.frame)
        self.background.photo = background_image
        self.background.place(x=0, y=0)

        # back button
        back_image = tk.PhotoImage(file='assets/back_button.png')
        self.backButton = tk.Button(image=back_image, master=self.frame, command=self.openMainFrame, highlightthickness = 0, bd = 0)
        self.backButton.photo = back_image
        self.backButton.place(x=100, y=555)

        # scan button
        scan_image = tk.PhotoImage(file='assets/small_scan.png')
        self.scanButton = tk.Button(image=scan_image, master=self.frame, command=self.scanPress, highlightthickness = 0, bd = 0)
        self.scanButton.photo = scan_image
        self.scanButton.place(x=300, y=555)

        # continue button
        continue_image = tk.PhotoImage(file='assets/solve_button.png')
        self.continueButton = tk.Button(image=continue_image, master=self.frame, command=self.openSudoku, highlightthickness = 0, bd = 0)
        self.continueButton.photo = continue_image
        self.continueButton.place(x=500, y=555)

        # image display
        self.imageDisplay = tk.Label(master=self.frame)
        self.imageDisplay.place(x=100, y=50)

        # pack frame
        self.frame.pack()
        self.frame.pack_propagate(0)

    def openSudoku(self):
        if(self.grid[0][0] != None):
            SudokuDisplay(self.master, self.grid)
            self.frame.pack_forget()

    def openMainFrame(self):
        self.isScanning = False # stops scanning
        MainFrame(self.master) # opens main frame
        self.frame.pack_forget() # closes the scanning window

    def scanPress(self):
        self.isScanning = True
        cam = cv2.VideoCapture(1) # creates VideoCapture object
        scanThread = Thread(target=self.scan, args=[cam]) # creates a new thread to run scan() so the main one can continue to monitor events
        scanThread.start() # starts the thread

    def scan(self, cam):
        while self.isScanning: # continues to run until the scan frame is closed
            cap, img = cam.read()  # read camera input
            img = img[50:550, 50:550]  # get only the top left 500x500 pixels
            noShadow = helpers.removeShadow(img)  # removes the shadow in the image

            grey = cv2.cvtColor(noShadow, cv2.COLOR_BGR2GRAY)  # convert to greyscale
            grey = cv2.GaussianBlur(grey, (5, 5), 1)  # blur
            ret, thresh = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY)  # create threshold image
            contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find the contours around the white spaces from the threshold
            contours = list(filter(helpers.filterContours, contours))  # filter out the contours that are too big or small to be squares

            cv2.drawContours(img, contours, -1, (255, 0, 0), 2) # draws the contours on the unedited image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img) # converts the cv2 image into a usable form
            img = ImageTk.PhotoImage(image=img)
            self.imageDisplay.configure(image=img) # replaces the label with the image
            self.imageDisplay.image = img

            if (len(contours) == 81): # if all 81 squares are detected, exits
                cam.release()
                self.isScanning = False
                corners = helpers.findCorners(contours)
                self.grid = helpers.getSudokuGrid(corners, thresh)


class SudokuDisplay():
    def __init__(self, master, grid=np.zeros((9, 9))):
        self.master = master
        self.frame = tk.Frame(width=700, height=700, master=self.master)

        # sets the background
        background_image = tk.PhotoImage(file='assets/sudoku_background.png')
        self.background = tk.Label(image=background_image, master=self.frame)
        self.background.photo = background_image
        self.background.place(x=0, y=0)

        # sets up the grid
        self.grid = grid
        self.gridFrame = tk.Frame(width=500, height=500, master=self.frame, bg='red')

        # back button
        back_image = tk.PhotoImage(file='assets/back_button.png')
        self.backButton = tk.Button(image=back_image, master=self.frame, command=self.toMenu, highlightthickness = 0, bd = 0)
        self.backButton.photo = back_image
        self.backButton.place(x=100, y=585)

        # submit button
        submit_image = tk.PhotoImage(file='assets/solve_button.PNG')
        self.submitButton = tk.Button(image=submit_image, master=self.frame, command=self.solve, highlightthickness = 0, bd = 0)
        self.submitButton.photo = submit_image
        self.submitButton.place(x=500, y=585)

        # finishes the frame and grid
        self.frame.pack()
        self.frame.pack_propagate(0)
        self.gridFrame.place(x=112, y=50) # 112, 100

        # populates the grid frame with numbers from the np array
        for i in range(9):
            for j in range(9):
                square = tk.Text(master=self.gridFrame, font=('Helvetica', 32), width=2, height=1)
                if(grid[i][j] != 0):
                    square.insert(1.0, str(grid[i][j]))
                square.grid(row=i, column=j)

    def toMenu(self):
        MainFrame(self.master)
        self.frame.pack_forget()

    def solve(self):
        # gets the numbers from the grid frame and converts them to np array
        for i in range(9):
            for j in range(9):
                number = self.gridFrame.grid_slaves(row=i, column=j)[0].get('1.0', 'end-1c')
                number = 0 if number == '' else number
                self.grid[i][j] = number
        # solves the square
        self.grid = self.grid.astype(int)
        solved = helpers.solve(self.grid)
        print(solved)
        # populates the grid frame with the np array if a solution was found
        if(solved):
            for i in range(9):
                for j in range(9):
                    number = self.grid[i][j]
                    number = 0 if number == '' else number
                    if(self.gridFrame.grid_slaves(row=i, column=j)[0].get('1.0', 'end-1c') == ''):
                        self.gridFrame.grid_slaves(row=i, column=j)[0].insert('1.0', number)


root = tk.Tk()
app = MainFrame(root)
root.mainloop()








