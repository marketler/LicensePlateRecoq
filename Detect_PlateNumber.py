import cv2
from cv2 import checkHardwareSupport
import imutils
from matplotlib.pyplot import text
import numpy as np
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'


def check_license(new_img):
    img = cv2.imread(new_img,cv2.IMREAD_COLOR)
    # cv2.imshow('1',img)
    # print(img.shape)

    h, w, channel = img.shape
    h2 = int(h/3)
    img = img[h2:int(2*h2), 0:w]
    # cv2.imshow('2',img)
    # print(img.shape)

    # img = cv2.resize(img, (1280, 360) )
    # cv2.imshow('3',img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 

    edged = cv2.Canny(gray, 200, 300) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in contours:
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
        # print(len(approx))
        # print(approx)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
        text = "No plate detected" 
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
        cv2.rectangle(img, (5, img.shape[0]-35), (520, img.shape[0]-5), (0,0,0), -1)
        cv2.putText(img, text= str(datetime.now()), org=(10, img.shape[0]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(img,img,mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        text = text.replace("\n", "")
        print("Detected license plate Number is:", text.replace("\n", ""))

    return text

    #     img = cv2.resize(img,(500,300))
    #     Cropped = cv2.resize(Cropped,(400,200))
    #     cv2.imshow('car',img)
    #     cv2.imshow('Cropped',Cropped)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#     cv2.imshow('Input', frame)

#     c = cv2.waitKey(1)
#     if c == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()





# print(check_license('Data\\308_oud.jpg'))