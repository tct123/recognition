import json
import sys
import threading
import tkinter as tk

from PIL import Image, ImageTk

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.quantization import mobilenet

# Initialize classification model
model = mobilenet.mobilenet_v2(pretrained=True, quantize=True)
model.eval()
# Image processing required for MobileNetV2
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# Load ImageNet classes
imagenetclasses = json.load(open("imagenet-simple-labels.json"))
MIN_SCORE = 0.5

# Initialize Tkinter
root = tk.Tk()
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
lmain = tk.Label(root)
lmain.grid()

# Initialize OpenCV
cap = cv2.VideoCapture(0)
# Check that we have camera access
if not cap.isOpened():
    lmain.config(
        text="Unable to open camera: please grant appropriate permission in Pydroid permissions plugin and relaunch.\nIf this doesn't work, ensure that your device supports Camera NDK API: it is required that your device supports non-legacy Camera2 API.",
        wraplength=lmain.winfo_screenwidth(),
    )
    root.mainloop()
    sys.exit(0)
else:
    # You can set the desired resolution here
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def refresh():
    global imgtk
    global curimage
    ret, frame = cap.read()
    if not ret:
        # Error capturing frame, try next time
        lmain.after(0, refresh)
        return
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w = lmain.winfo_screenwidth()
    h = lmain.winfo_screenheight()
    cw = cv2image.shape[0]
    ch = cv2image.shape[1]
    cw, ch = ch, cw
    if (w > h) != (cw > ch):
        cw, ch = ch, cw
        cv2image = cv2.rotate(cv2image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    w = min(cw * h / ch, w)
    h = min(ch * w / cw, h)
    # Leave some space for label in landscape
    if h == lmain.winfo_screenheight():
        w *= 0.9
        h *= 0.9
    w, h = int(w), int(h)
    # At this point our image is ready for classification
    curimage = cv2image
    # Resize to fill the whole screen
    cv2image = cv2.resize(cv2image, (w, h), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.configure(image=imgtk)
    lmain.update()
    lmain.after(0, refresh)


# Classification will be done in a separate thread
curimage = None


def classify():
    global curimage
    while True:
        if curimage is not None:
            tmp = curimage
            input_tensor = preprocess(Image.fromarray(tmp))
            input_batch = input_tensor.unsqueeze(0)
            with torch.no_grad():
                output = model(input_batch)
            rx = torch.nn.functional.softmax(output[0], dim=0)
            maxid = int(np.argmax(rx))
            if rx[maxid] < MIN_SCORE:
                txt = "No object detected"
            else:
                txt = "Detected the following object: " + imagenetclasses[maxid]
            lmain.config(
                text=txt, wraplength=lmain.winfo_screenwidth(), compound=tk.BOTTOM
            )


refresh()
threading.Thread(target=classify).start()
root.mainloop()
