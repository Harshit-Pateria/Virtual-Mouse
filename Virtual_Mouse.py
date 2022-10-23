import cv2
import numpy as np
import mediapipe as mp
import time
import Hand_Tracker as ht
import pyautogui
pyautogui.FAILSAFE = False
# Variables
width = 640            # Width of Camera
height = 480            # Height of Camera
s_width, s_height = pyautogui.size()
prev_x, prev_y = 0,0
x, y, x2, y2 = 0,0,0,0
s = 8                    # smoothening

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)           # Adjusting size
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
flag = 0
move = 0

d = ht.HandDetector()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = d.find_hand(img)
    pos = d.find_pos(img)
    

    if len(pos) != 0:
        fingers = d.fingers_up()
        x1,y1 = pos[9][:2]
        #COMMANDS
        if fingers[1] == 1 and fingers.count(1) == 1 and flag == 0:
            pyautogui.click()   
        if fingers.count(1) == 5:                     # reset cursor start points
            move = 1
        if fingers[0] == 1 and fingers[4] == 1 and fingers.count(1) == 2 and flag == 0:    # Play/Pause
            pyautogui.press('playpause')
            flag = 1
        if fingers.count(1) == 0:          # Reset flag and cursor move   
            flag = 0
            x2 = np.interp(x1, (0,width), (0,s_width))           #interpolate
            y2 = np.interp(y1, (0,height), (0,s_height))
            x = x2 - (x2 - prev_x) / s                       #smoothen
            y = y2 - (y2 - prev_y) / s
            if move == 1:
                prev_x, prev_y = x, y
                move = 0
            pyautogui.moveRel((x - prev_x) ,(y - prev_y))
            prev_x, prev_y = x2,y2
    
    cv2.imshow("Image", img)
        
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
