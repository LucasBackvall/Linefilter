from PIL import Image, ImageFilter
import numpy as np
import cv2
import time
import threading

# Line filter testing
# This program implements a simple edge-detection algorithm

# Usage:
# Simply run this with python3 on a device with a webcam.
# Press q to exit
# Press a number between 1-9 to select resolution of filter

# Constants
debug = False

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

def lineFilter(im, size=5):
    
    width, height = im.shape
    
    owidth, oheight = width-2*size -1, height-2*size -1
    oim = np.zeros((owidth, oheight), np.uint8)
    
    filt = np.ones((2*size+1, 2*size+1), np.uint8)
    filt = np.multiply(filt, 1/((size*2+1)**2))
    filt.itemset((size, size), -1)
    
    for y in range(oheight):
        for x in range(owidth):
            val = 0
            pixel = im.item(x+size, y+size)
            
            m = im[x:x+2*size+1, y:y+2*size+1]
            res = np.multiply(m, filt)
            val = np.sum(res)
            
            if val > 2*size:
                pixel = 255
            else:
                pixel = 0
            oim.itemset((x, y), np.uint8(pixel))
    return oim

class shared():
    
    def __init__(self, frame, f):
        self.frame = frame
        self.f = f
        self.n = 2
        self.loop = True



# Setup webcam
try:
    cam = cv2.VideoCapture(0)
except:
    cam = cv2.VideoCapture(input("Capture device number:\n>>"))

# First frame
_, frame = cam.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow("Input", frame)
cv2.moveWindow("Input", 800, 0)

# Create shared object
data = shared(frame, 0.5)

def outputThread(data):
    
    while data.loop:
        
        iframe = cv2.resize(data.frame, (0,0), fx=data.f, fy=data.f)
        
        # Process
        pframe = lineFilter(iframe, data.n)
        
        # Output
        pframe = cv2.resize(pframe, (0,0), fx=1/data.f, fy=1/data.f)
        cv2.imshow("Output", pframe)
        
        time.sleep(1/24)

tOutput = threading.Thread(target=outputThread, args=[data])
tOutput.start()

counter = 0

dbg("start")

while True:
    
    # 24 fps
    time.sleep(1/24)
    
    # Update frame
    dbg(str(counter) + " update frame")
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Input", frame)
    key = cv2.waitKey(1)
    data.frame = frame
    
    dbg("frame updated")
    # Set varialbe n
    if key > ord("0") and key <= ord("9"):
        data.f = (key - ord("0"))/10
    
    # exit condition
    if key == ord("q"):
        data.loop = False
        break

tOutput.join()

