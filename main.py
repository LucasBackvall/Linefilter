from PIL import Image, ImageFilter
import numpy as np
import cv2
import time, threading, sys

# Line filter testing
# This program implements a simple edge-detection algorithm

# Usage:
# Simply run this with python3 on a device with a webcam.
# Press q to exit
# Press a number between 1-4 to select resolution of image
# Press a number between 5-9 to select size of filter
# Press + or - to change threshold value
# Press t to toggle showing the input in the background

# Constants
debug = False
#debug = True

# Debug function
def dbg(inp):
    global debug
    if debug:
        print("Debug: " + inp)

def offset(value):
    return value-128

def quantize(value, cutoff=128):
    if value >= cutoff:
        return 255
    else:
        return 0

def inverse(value, cutoff=128):
    if value < cutoff:
        return 255
    else:
        return 0

def elemCalc(iMatrix, preCalculatedMatrix, oMatrix, size, width, y, threshold):
    for x in range(width):
        
        # Centre element
        elem = iMatrix.item(x+size, y+size)
        
        # Fetch pre-calculated floating point digits
        filterMatrix = preCalculatedMatrix[x:x+2*size+1, y:y+2*size+1]
        
        val = np.sum(filterMatrix) - elem
        
        if val > threshold:
            oMatrix.itemset((x, y), np.uint8(0))
        else:
            oMatrix.itemset((x, y), np.uint8(1))

def lineFilter(iMatrix, size=5, threshold=10):
    
    # This function creates a 2D filter matrix with
    # width and height 2*size+1
    # The elements of the matrix are 1/(2*size+1)^2,
    # then we add -1 to the centre element.
    
    width, height = iMatrix.shape
    
    oWidth, oHeight = width-2*size -1, height-2*size -1
    oMatrix = np.zeros((oWidth, oHeight), np.uint8)
    
    preCalculatedMatrix = np.multiply(iMatrix, 1/((size*2+1)**2))
    
    threads = []
    
    for y in range(oHeight):
        t = threading.Thread(target=elemCalc, args=[iMatrix, preCalculatedMatrix, oMatrix, size, oWidth, y, threshold])
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    return oMatrix

# Shared variables class
class shared():
    
    def __init__(self, frame, scale, size, threshold):
        # Input frame
        self.frame = frame
        # Layered frame
        width, height = frame.shape 
        w, h = width-2*size -1, height-2*size -1
        self.layer = np.zeros((w, h), np.uint8)
        # Scale of input image to be used in processing (0-1)
        self.scale = scale
        # Size of filter matrix to be used by processing
        self.size = size
        # Threshold for filter cutoff. This is basically how to define what's
        # an edge, and what's just noise.
        self.threshold = threshold
        # Thread control variable. Set to false to kill all extra threads.
        self.loop = True


# Setup webcam

dbg("Setup webcam begin.")
try:
    # Try getting first frame
    cam = cv2.VideoCapture(0)
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noWebcam = False
except:
    noWebcam = True

while noWebcam:
    cv2.destroyAllWindows()
    inp = input("Enter Q to exit program. Otherwise enter capture device number:\n>>")
    if inp == "Q" or inp == "q":
        sys.exit()
    try:
        # Try getting first frame
        cam = cv2.VideoCapture(int(inp))
        _, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noWebcam = False
    except:
        print("Not a valid capture device number.")

# Create shared object
frame = cv2.resize(frame, (0,0), fx=1, fy=1)
data = shared(frame, 1, 1, 5)

def outputThread(data):
    
    while data.loop:
        
        # Process
        data.layer = lineFilter(data.frame, data.size, data.threshold)
        layer = np.multiply(data.layer, 255)
        if debug:
            cv2.imshow("Layer", cv2.resize(layer, (0,0), fx=1/data.scale, fy=1/data.scale))
        

tOutput = threading.Thread(target=outputThread, args=[data])
tOutput.start()

showInput = True

time.sleep(0.5)
dbg("Loop begin")

while True:
    
    # 24 fps
    time.sleep(1/24)
    
    # Update input frame
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (0,0), fx=data.scale, fy=data.scale)
    data.frame = frame
    width, height = frame.shape
    if showInput:
        frame = frame[data.size:width-data.size-1, data.size:height-data.size-1]
    else:
        frame = np.ones((width-data.size*2-1, height-data.size*2-1), np.uint8)
        frame = np.multiply(frame, 255)
    if debug:
        cv2.imshow("Input", cv2.resize(frame, (0,0), fx=1/data.scale, fy=1/data.scale))
    
    # Add layer
    if data.layer.shape == frame.shape:
        frame = np.multiply(data.layer, frame)
    else:
        dbg("Layer size mismatch.")
    
    # Show frame
    cv2.imshow("Display", cv2.resize(frame, (0,0), fx=1/data.scale, fy=1/data.scale))
    key = cv2.waitKey(1)
    
    # Set varialbe f
    if key >= ord("1") and key <= ord("4"):
        data.scale = (key - ord("0"))/4

    # Set varialbe n
    elif key >= ord("5") and key <= ord("9"):
        data.size = key - ord("0") - 4
    
    elif key == ord("+"):
        data.threshold += 1
        print("Increasing threshold do %i" % data.threshold)

    elif key == ord("-"):
        data.threshold -= 1
        print("Decreasing threshold do %i" % data.threshold)

    elif key == ord("t"):
        showInput = not showInput
    
    # exit condition
    elif key == ord("q"):
        data.loop = False
        break

tOutput.join()
cv2.destroyAllWindows()

