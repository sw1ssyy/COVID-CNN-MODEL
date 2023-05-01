import tkinter as ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import ttkthemes as tkthemes
from keras.models import load_model
import numpy as np
import cv2



root = tkthemes.ThemedTk(theme="Adapta")
root.title("CoronaVision")
root.resizable(False,False)
root.iconbitmap("Logo.ico")
root.eval('tk::PlaceWindow . center')

canvas = ttk.Canvas(root, width=400, height=400)
canvas.pack(side="top")

filename_label = ttk.Label(root, text="", font=("Helvetica", 11))
filename_label.pack(side="top")

Diagnosis_label = ttk.Label(root, text="", font=("Helvetica", 11))
Diagnosis_label.pack(side="top")

#Import CNN COVID Model
Model = load_model("COVID_CNN_MODEL")

#File that Claassifed Results will be held
ResultsFile = open("Results.txt", "a")
# Function for opening processing and displaying the image
def open_image():
    
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.png")])
    # For displaying ct scan
    img = Image.open(file_path)
    img = img.resize((256, 256))

 # Convert the image to a 4D tensor by adding a batch dimension
    images = []
    predimg = cv2.imread(file_path)
    predimg = cv2.resize(predimg, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    images.append(np.array(predimg))

    
    testimage = np.expand_dims(images[0], axis=0)
    print(np.shape(testimage))
    #(1, 256, 256, 3)

        # Pass the image through the loaded model
    prediction = Model.predict(testimage)

    if prediction >= 0.5:
        prediction = 1

    elif prediction < 0.5:
       prediction = 0
       
       
       #IF prediction is NONCOVID
    if prediction == 1:
        Diagnosis_label.config(text = "Prediction: NONCOVID", font=("Arial", 12))
        ResultsFile.write("{}\n".format(str(file_path.split("/")[-1]) + ": " + "NONCOVID"))
       #IF prediction is COVID
    elif prediction == 0:
       Diagnosis_label.config(text = "Prediction: COVID", font=("Arial", 12))
       ResultsFile.write("{}\n".format(file_path.split("/")[-1] + ": " + "COVID "))
      
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(200, 250,  image=img_tk, anchor="center")
    canvas.image = img_tk
    filename_label.config(text= "Image: " + file_path.split("/")[-1])


button = ttk.Button(root, text="Open Image", command=open_image)
button.pack(side="bottom")


root.mainloop()
