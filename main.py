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

def lineFilter(im, size=5):
    
    width, height = im.shape
    
    owidth, oheight = width-2*size -1, height-2*size -1
    oim = np.zeros((owidth, oheight), np.uint8)
    
    # Create a size*size matrix where center element is -1
    # and the rest of the elemets is 1/(size^2)
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

# Shared variables class
class shared():
    
    def __init__(self, frame, f):
        # Input frame
        self.frame = frame
        # Scale of input image to be used in processing (0-1)
        self.f = f
        # Size of filter matrix to be used by processing
        self.n = 2
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



# Show first frame
cv2.imshow("Input", frame)
cv2.moveWindow("Input", 800, 0)

# Create shared object
data = shared(frame, 0.4)

def outputThread(data):
    
    while data.loop:
        
        # Scale down input frame to a factor f or original size
        iframe = cv2.resize(data.frame, (0,0), fx=data.f, fy=data.f)
        
        # Process
        pframe = lineFilter(iframe, data.n)
        
        # Output, scaled up again by a factor of 1/f
        pframe = cv2.resize(pframe, (0,0), fx=1/data.f, fy=1/data.f)
        cv2.imshow("Output", pframe)
        
        time.sleep(1/24)

tOutput = threading.Thread(target=outputThread, args=[data])
tOutput.start()

counter = 0

dbg("Loop begin")

while True:
    
    # 24 fps
    time.sleep(1/24)
    
    # Update input frame
    dbg("Update input frame")
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Input", frame)
    key = cv2.waitKey(1)
    data.frame = frame
    
    dbg("Input frame updated")
    # Set varialbe f
    if key >= ord("1") and key <= ord("4"):
        data.f = (key - ord("0"))/10

    # Set varialbe n
    if key >= ord("5") and key <= ord("9"):
        data.n = key - ord("0") - 4
    
    # exit condition
    if key == ord("q"):
        data.loop = False
        break

tOutput.join()
cv2.destroyAllWindows()

