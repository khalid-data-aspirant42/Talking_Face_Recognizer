# A face recognizer that talks!

### Task: to build a basic model that recognize multiple faces and does voice confirmation after detection

### Steps:
'''1. Getting user id with name.
2. Taking pics of new user and collecting in a database.
3. Training the model with haarcascade inbuilt file by creating numpy array.
4.  Finally, in the recognision part, face detected and compared with the trained model. 
The closed ones is called up with probability of user presence.'''

import cv2  #openCV - open source Computer Vision library
import os   #Operating System module to interaction with local environment
import pyttsx3  #text-to-speech conversion library
from functools import reduce  #to get unique list

# photoCapture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# creating a classifier model using haarcascade file to distinguise between face and non-face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# creating voice
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# For each person, enter one numeric face id
engine.say("face detection is processing. please enter your username following your face id")
engine.runAndWait()
user_name = str(input('Enter your 6 digit username following face_id(e.g.,1khalid,2aditya)'))

entries = os.listdir('C:/Users/Administrator/Desktop/computer vision/final_model/database/')
t_id = [elem[8] for elem in entries]
t_name = [elem[1:7] for elem in entries[::-1]]
def unique(list1):
    ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
    return ans
u_id = unique(t_id) 
u_name = unique(t_name[::-1])

if user_name[0] in u_id or user_name in u_name:
    print('User already exist')
    engine.say("user data found")
    engine.say("please look into the camera for identification")
    engine.runAndWait()
else:
    engine.say("new user detected")
    engine.say("camera opening up for face submission")
    engine.runAndWait()
    engine.runAndWait()
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    while(True):

        ret, img = cam.read()
        # img = cv2.flip(img, -1) - flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("C:/Users/Administrator/Desktop/computer vision/final_model/database/" + user_name + '.'+ str(user_name[0]) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
             break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    engine.say("data successfully submitted")
    engine.say("please wait while model is trained")
    engine.runAndWait()
   
    cam.release()
    cv2.destroyAllWindows()
    
    '''Training the data'''
    import cv2  
    import numpy as np
    from PIL import Image
    import os

    # Path for face image database
    path = 'C:/Users/Administrator/Desktop/computer vision/final_model/database/'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('C:/Users/Administrator/Desktop/computer vision/final_model/trained_model/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    engine.say("model is successfully trained")
    engine.say("look into camera for identity confirmation")
    engine.runAndWait()


"""Testing(Recognition)"""
import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/Administrator/Desktop/computer vision/final_model/trained_model/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None'] + u_name

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    # img = cv2.flip(img, -1) - Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 20,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "No Identity"
            confidence = "  {0}%".format(round(100 - confidence))
            engine.say('Unknown face identified')
            engine.runAndWait()
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

cam.release()
engine.say(f'Thank you {user_name[1:]}. Identification confirmed')
engine.runAndWait()
cv2.destroyAllWindows()
