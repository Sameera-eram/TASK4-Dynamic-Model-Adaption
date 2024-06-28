import cv2
import numpy as np
from keras.api.models import model_from_json
from keras.src.optimizers import Adam
import tkinter as tk
from PIL import Image, ImageTk


json_file = open("model_a.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model_weights.weights.h5")


haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


root = tk.Tk()
root.title("Real-time Facial Expression Recognition")


frames = []
prediction_label = None

def extract_features(frames):
    features = []
    for frame in frames:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        if len(face) == 1:
            (p, q, r, s) = face[0]
            face_img = frame_gray[q:q+s, p:p+r]
            face_img = cv2.resize(face_img, (48, 48))
            features.append(face_img)
        else:
            features.append(np.zeros((48, 48), dtype=np.uint8))
    if len(features) != 5:
        print("Error: Detected less than 5 faces in the sequence.")
        return None
    features = np.array(features)
    features = features.reshape(1, 5, 48, 48, 1) 
    features = features / 255.0  
    return features

def process_frame():
    global prediction_label
    _, frame = webcam.read()
    frames.append(frame)
    if len(frames) > 5:
        frames.pop(0)

    features = extract_features(frames)

    if features is not None:
        prediction = model.predict(features)
        prediction_label = labels[np.argmax(prediction)]
        label_var.set("Predicted emotion: " + prediction_label)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 480))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)
    panel.after(10, process_frame)

def correct_feedback():
    label_var.set("Thank you for your feedback!")

def incorrect_feedback():
    feedback_window = tk.Toplevel(root)
    feedback_window.title("Provide Correct Emotion")

    tk.Label(feedback_window, text="Please select the correct emotion:").pack(pady=10)
    
    emotion_var = tk.StringVar(value="neutral")
    for idx, emotion in labels.items():
        tk.Radiobutton(feedback_window, text=emotion, variable=emotion_var, value=emotion).pack(anchor=tk.W)
    
    def submit_feedback():
        correct_emotion = emotion_var.get()
        update_model(frames, correct_emotion)
        feedback_window.destroy()
    
    tk.Button(feedback_window, text="Submit", command=submit_feedback).pack(pady=10)

def update_model(frames, correct_emotion):
    global model
    features = extract_features(frames)
    if features is not None:
        correct_label = [key for key, value in labels.items() if value == correct_emotion][0]
        correct_label_one_hot = np.zeros((1, 7))
        correct_label_one_hot[0, correct_label] = 1
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(features, correct_label_one_hot, epochs=1, batch_size=1, verbose=0)
        label_var.set("Model updated with the correct emotion: " + correct_emotion)

label_var = tk.StringVar()
label_var.set("Predicted emotion: ")
label = tk.Label(root, textvariable=label_var, font=("Helvetica", 16))
label.pack(pady=20)

panel = tk.Label(root)
panel.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

correct_button = tk.Button(button_frame, text="Correct", command=correct_feedback)
correct_button.grid(row=0, column=0, padx=10)

incorrect_button = tk.Button(button_frame, text="Incorrect", command=incorrect_feedback)
incorrect_button.grid(row=0, column=1, padx=10)

webcam = cv2.VideoCapture(0)

process_frame()

root.mainloop()

webcam.release()
cv2.destroyAllWindows()
