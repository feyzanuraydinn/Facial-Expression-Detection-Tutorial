import cv2
import numpy as np
from keras.models import load_model

model = load_model('model_file.h5')

video = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        class_label = labels_dict[label]
        text_width, _ = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame, class_label, (x + (w - text_width) // 2, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    percentages = [f"{labels_dict[i]}: {int(val*100)}%" for i, val in enumerate(result[0])]
    text_position_percent = (frame.shape[1] - 430, 30)
    for i, percent in enumerate(percentages):
        text_size = cv2.getTextSize(percent, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        x_pos = text_position_percent[0] + (400 - text_size[0])
        y_pos = text_position_percent[1] + 25 * (i+1)
        cv2.putText(frame, percent, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
