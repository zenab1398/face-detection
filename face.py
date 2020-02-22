import cv2
#start camera
cap=cv2.VideoCapture(0)
#loading haar file
face_detect=cv2.CascadeClassifier('face.xml')
eye_detect=cv2.CascadeClassifier('eye1.xml')
#smile_detect=cv2.CascadeClassifier('smile.xml')

print(cap.isOpened())

while cap.isOpened():
    status,frame=cap.read()
    #converting color image to gray scale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #now using haar functions
    face=face_detect.detectMultiScale(gray)
    print(face)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        onlyface=frame[y:y+h,x:x+w]
        eye=eye_detect.detectMultiScale(onlyface)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(onlyface,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        #smile=smile_detect.detectMultiScale(onlyface)
        #for (sx,sy,sw,sh) in smile:
         #   cv2.rectangle(onlyface,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)

    cv2.imshow('face',frame)
    cv2.imshow('face',onlyface)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


