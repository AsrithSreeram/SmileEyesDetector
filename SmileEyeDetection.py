import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')


webcam = cv2.VideoCapture(0)

while True: 
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_detector.detectMultiScale(grayscaled_frame)


    for (x, y, width, height) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (100, 200, 50), 2)

        smileeyes_in_face = frame[y:y+height, x:x+width]
        #x, y = top left point of face
        #x+width, y+height = bottom right point of face
        #this will make it easy to draw a rectangle because we have the top left and bottom right points
        grayscaled_face = cv2.cvtColor(smileeyes_in_face, cv2.COLOR_BGR2GRAY)

        smile_coordinates = smile_detector.detectMultiScale(grayscaled_face, scaleFactor=1.9, minNeighbors=25) #manually tuning parameters for greater accuracy

        eyes_coordinates = eye_detector.detectMultiScale(grayscaled_face, scaleFactor=1.3, minNeighbors=20) #manually tuning parameters for greater accuracy


        for (x_, y_, width_, height_) in smile_coordinates:
            cv2.rectangle(smileeyes_in_face, (x_, y_), (x_+width_, y_+height_), (50, 100, 200), 2)

        for (x__, y__, width__, height__) in eyes_coordinates:
            cv2.rectangle(smileeyes_in_face, (x__, y__), (x__+width__, y__+height__), (50, 100, 200), 2)

        if len(smile_coordinates) > 0: #checking if there is a smile and eyes on the face
            cv2.putText(frame, 'Smiling', (x, y+height+40), fontScale=3,
            fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

        if len(eyes_coordinates) > 0: #checking if there is a smile and eyes on the face
            cv2.putText(frame, 'Eyes', (x, y+height+70), fontScale=3,
            fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))



    cv2.imshow("Smile Detector", frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()
cv2.destroyAllWindows()
print("Complete")
