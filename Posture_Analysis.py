#Import Statements

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.optimizers import Adam

#Tiny-VGG Model

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (96, 96, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (3, 3)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(64, (3, 3), activation = 'sigmoid'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'sigmoid'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 4, activation = 'softmax'))

#Compiling the Model

opt = Adam(lr = 1e-5)
classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

#Importing the Dataset 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
    
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('cnn/train',
                                                 target_size = (96, 96),
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory('cnn/test',
                                            target_size = (96, 96),
                                            batch_size = 32)

#Fitting the Model

from keras.callbacks import History
history = History()

classifier.fit_generator(training_set,
                         steps_per_epoch = 2076,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 145,
                         callbacks=[history])


# Taking Input from WebCam

from keras.preprocessing import image
import numpy as np
import cv2

cap = cv2.VideoCapture(0) #Capturing Video from Webcam
while(True):
    res, pic = cap.read() #Capturing each Frame
    cv2.imwrite('pic.jpg', pic)
    
    test_image = image.load_img('pic.jpg', target_size=(96, 96))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    res = classifier.predict(test_image) #Predicting the Frame
    
    if (np.argmax(res[0]) == 0):
        cv2.putText(pic,('Crunches - '+ str(res[0][0])),
                    (32,36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),2)
        exe = cv2.imread('crunche.jpg') 
        cv2.imshow('crunches', exe) #Displaying the correct Image
    
    if (np.argmax(res[0]) == 1):
        cv2.putText(pic,'Relaxed',
                    (32,36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),2)
    
    if (np.argmax(res[0]) == 2):
        cv2.putText(pic,('Pushup - '+ str(res[0][2])),
                    (32,36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),2)
        exe = cv2.imread('pushup.jpg')
        cv2.imshow('pushup', exe)
    
    if (np.argmax(res[0]) == 3):
        cv2.putText(pic,('Squat - '+ str(res[0][3])),
                    (32,36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),2)
        exe = cv2.imread('squat.jpg')
        cv2.imshow('squat', exe)
    
    cv2.imshow('pic', pic) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #Waiting until the user presses 'q' on keyboard
        break
cap.release()
cv2.destroyAllWindows()



