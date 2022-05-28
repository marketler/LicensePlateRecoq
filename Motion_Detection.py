from Detect_PlateNumber import check_license
import numpy as np
import cv2, time
from datetime import datetime
import csv
import os

# Open csv-file
with open('Time_of_movements.csv', 'a', newline='', encoding='utf-8') as file:

    # Capture video
    video = cv2.VideoCapture(0)
    check, frame = video.read()

    # Create avgs for rolling average
    avg1 = np.float32(frame)
    
    # Infinite while loop to treat stack of image as video
    while True:
        # Reading frame(image) from video
        check, frame = video.read()

        h, w, c = frame.shape

        frame = frame[0:h, 0:w]

        # Edit rolling average to account for small daily changes in lighting etc.
        cv2.accumulateWeighted(frame,avg1,0.1) # Change the int to alter speed of average adjustment
        res1 = cv2.convertScaleAbs(avg1)
        cv2.imshow('avg1',res1)
        
        # Convert average to grayscale
        gray_res = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
        gray_res = cv2.GaussianBlur(gray_res, (21, 21), 0)

        static_back = gray_res
    
        # Converting current frame to gray_scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Converting gray scale image to GaussianBlur
        # so that change can be find easily
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        

        # Difference between static background
        # and current frame(which is GaussianBlur)
        diff_frame = cv2.absdiff(static_back, gray)
    
        # If change in between static background and
        # current frame is greater than 30 it will show white color(255)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    
        # Finding contour of moving object
        cnts,_ = cv2.findContours(thresh_frame.copy(),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
    
            (x, y, w, h) = cv2.boundingRect(contour)
            # making green rectangle around the moving object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # call platenumber detection
            detected_license = check_license('Data\\308_oud_2.jpg')

            # print timestamp on capture
            cv2.rectangle(frame, (5, frame.shape[0]-65), (520, frame.shape[0]-5), (0,0,0), -1)
            cv2.putText(frame, text= str(datetime.now()), org=(10, frame.shape[0]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

            # print detected license on capture
            cv2.putText(frame, text=detected_license, org=(10, frame.shape[0]-40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
            
            # Save motion capture in filesystem
            filename = "motion_" + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ".jpg"
            # filename = "motion_1.jpg"
            cv2.imwrite(os.path.join('Data/Captures', filename), frame)

            # Write motion caputre metadata to csv
            metadata_set = [datetime.now(), filename, detected_license]
            
            # write 
            writer = csv.writer(file)
            writer.writerow(metadata_set)

            # Show motion capture
            cv2.imshow('motion_cap', frame)

            if detected_license == "13-KNB-6":
                # insert code that will trigger gate open
                print("match")
            else:
                # (optional) insert code that will trigger alarm
                print("no match")
        
            time.sleep(0.2)
    
    
        # Displaying image in gray_scale
        cv2.imshow("Gray Frame", gray)
    
        # Displaying the difference in currentframe to
        # the staticframe(very first_frame)
        cv2.imshow("Difference Frame", diff_frame)
    
        # Displaying the black and white image in which if
        # intensity difference greater than 30 it will appear white
        cv2.imshow("Threshold Frame", thresh_frame)
    
        # Displaying color frame with contour of motion of object
        cv2.imshow("Color Frame", frame)
    
        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            break
    
    file.close()
    video.release()
    
    # Destroying all the windows
    cv2.destroyAllWindows()