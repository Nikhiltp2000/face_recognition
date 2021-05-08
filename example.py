import cv2

trained_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('photo.jpg')//input the path of the photo
 

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face.detectMultiScale(grayscaled_img)

#(x, y, w, h) = face_coordinates[0]
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 10)



print(face_coordinates)




cv2.imshow('nikhil' , img)

cv2.waitKey()
