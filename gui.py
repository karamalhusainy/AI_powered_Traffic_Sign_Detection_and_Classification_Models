import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
#load the trained model to classify sign
from keras.models import load_model
import os
model = load_model("traffic_classifier1.h5")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#dictionary to label all traffic signs class.
classes = { 0:"No Entry", 1:"Hump", 2:"Stop", 3:"Pedestrian Cross", 4:"No Stop", 5:"Give Way", 6:"Pass Either"}


#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (60, 60))
    image = np.expand_dims(image, axis=0)
    # pred = model.predict_classes([image])[0]
    pred = model.predict(image)
    classes_x = np.argmax(pred, axis=1)[0]
    # print(classes_x , pred)
    sign = classes[classes_x]
    print(sign, classes_x)
    label.configure(foreground='#011638', text=sign + str(classes_x))

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()