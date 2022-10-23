import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=1,modelComp = 1,  detectionCon=0.7, trackCon=0.85):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, self.detectionCon, self.trackCon)
        
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        
    def find_hand(self, img, draw=True):                                             # Finds hand in frame
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_lms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def find_pos(self, img, draw = True):
        h,w,c = img.shape
        
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):
                px, py = int(lm.x*w), int(lm.y*h)
                self.lm_list.append([px, py, id])

        return self.lm_list
    
    def fingers_up(self):
        fingers = []

        if self.lm_list[self.tipIds[0]][0] < self.lm_list[self.tipIds[0] - 1][0] < self.lm_list[0][0] or self.lm_list[self.tipIds[0]][0] > self.lm_list[self.tipIds[0] - 1][0] > self.lm_list[0][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lm_list[self.tipIds[id]][1] < self.lm_list[self.tipIds[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    

def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)

    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        
        # print fps 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255, 255), 3) 

        img = detector.find_hand(img)
        pos = detector.find_pos(img)
        if len(pos) != 0:
            fingers = detector.fingers_up()
            print(pos[4], fingers.count(1))
        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

        

if __name__ == "__main__":
    main()