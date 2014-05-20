import cv
import time
from PIL import Image
 
def DetectFace(image, faceCascade):
 
    min_size = (20,20)
    image_scale = 2
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
 
    # Allocate the temporary images
    grayscale = cv.CreateImage((image.width, image.height), 8, 1)
    smallImage = cv.CreateImage(
            (
                cv.Round(image.width / image_scale),
                cv.Round(image.height / image_scale)
            ), 8 ,1)
 
    # Convert color input image to grayscale
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
 
    # Scale input image for faster processing
    cv.Resize(grayscale, smallImage, cv.CV_INTER_LINEAR)
 
    # Equalize the histogram
    cv.EqualizeHist(smallImage, smallImage)
 
    # Detect the faces
    faces = cv.HaarDetectObjects(
            smallImage, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )
 
    # If faces are found
    if faces:
        for ((x, y, w, h), n) in faces:
            # the input to cv.HaarDetectObjects was resized, so scale the
            # bounding box of each face and convert it to two CvPoints
            pt1 = (int(x * image_scale), int(y * image_scale))
            pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

	    #fourcc = cv.CV_FOURCC('M','P','1','V')
	    #fourcc = cv.CV_FOURCC('H','F','Y','U') # HuffYUV
	    #fourcc = cv.CV_FOURCC('D','R','A','C') # BBC Dirac
	    #fourcc = cv.CV_FOURCC('X','V','I','D') # MPEG-4 Part 2
	    #fourcc = cv.CV_FOURCC('X','2','6','4') # MPEG-4 Part 10 (aka. H.264 or AVC)
	    #fourcc = cv.CV_FOURCC('M','P','1','V') # MPEG-1 video
	    #cvw = cv.CreateVideoWriter("testing.avi", fourcc, 1.0, (640,480), 0)
	    #cv.WriteFrame(cvw, image) 
 
    return image
 
 

capture = cv.CaptureFromCAM(0)
cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
#capture = cv.CaptureFromFile("test.avi")
 
#faceCascade = cv.Load("haarcascades/haarcascade_frontalface_default.xml")
#faceCascade = cv.Load("haarcascades/haarcascade_frontalface_alt2.xml")
faceCascade = cv.Load("haarcascade_frontalface_alt.xml")
#faceCascade = cv.Load("haarcascades/haarcascade_frontalface_alt_tree.xml")
 
while (cv.WaitKey(15)==-1):
    img = cv.QueryFrame(capture)
    image = DetectFace(img, faceCascade)
    cv.ShowImage("face detection test", image)
 
cv.ReleaseCapture(capture)
