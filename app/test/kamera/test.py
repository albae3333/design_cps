import cv2
import time

Kamera_NR = 0

cap = cv2.VideoCapture(Kamera_NR)

zeit = int(time.time())

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    if zeit != int(time.time()):

        # Display the resulting frame

        cv2.imshow('frame',frame)

        zeit = int(time.time())
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()