from tkinter import *
from tkinter import filedialog
import tkinter as tk
from click import command
from PIL import Image
from PIL import ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageUtils as utils

global img

root = Tk()

def change_image():
    global img
    fname = filedialog.askopenfilename(
        filetypes=(("JPEG files", "*.jpg"), ("All files", "*")))
    if (fname != None):
        img = cv2.imread(fname)
        imagePIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(imagePIL)
        pic = canvas.create_image(0,0,anchor=NW,image=photo)
        canvas.itemconfig(pic,img[0])
    return

def greyscale():
    greyimg = utils.greyscale(img=img)
    plt.imshow(greyimg)
    plt.show()
def big():
    image = np.array(utils.big(img=img), dtype=np.uint8)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = Image.fromarray(RGB_img)
    photo = ImageTk.PhotoImage(imagePIL)
    pic = canvas.create_image(0,0,anchor=NW,image=photo)
    canvas.itemconfig(pic,img[0])
    plt.imshow(RGB_img)
    plt.show()
def small():
    image = np.array(utils.small(img=img), dtype=np.uint8)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = Image.fromarray(RGB_img)
    photo = ImageTk.PhotoImage(imagePIL)
    pic = canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.itemconfig(pic, image[0])
    plt.imshow(RGB_img)
    plt.show()

def brighter():
    global img
    image = np.array(utils.brighterkali(img=img), dtype=np.uint8)
    img = image
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = Image.fromarray(RGB_img)
    photo = ImageTk.PhotoImage(imagePIL)
    pic = canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.itemconfig(pic, image[0])
    plt.imshow(RGB_img)
    plt.show()
def darker():
    global img
    image = np.array(utils.darkerbagi(img=img), dtype=np.uint8)
    img = image
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = Image.fromarray(RGB_img)
    photo = ImageTk.PhotoImage(imagePIL)
    pic = canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.itemconfig(pic, image[0])
    plt.imshow(RGB_img)
    plt.show()
def geserkanan():
    global img
    image = np.array(utils.geserkanan(img=img), dtype=np.uint8)
    img = image
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = Image.fromarray(RGB_img)
    photo = ImageTk.PhotoImage(imagePIL)
    pic = canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.itemconfig(pic, image[0])
    plt.imshow(RGB_img)
    plt.show()

def geserkiri():
    global img
    image = np.array(utils.geserkiri(img=img), dtype=np.uint8)
    img = image
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = Image.fromarray(RGB_img)
    photo = ImageTk.PhotoImage(imagePIL)
    pic = canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.itemconfig(pic, image[0])
    plt.imshow(RGB_img)
    plt.show()
def geseratas():
    global img
    image = np.array(utils.geseratas(img=img), dtype=np.uint8)
    img = image
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = Image.fromarray(RGB_img)
    photo = ImageTk.PhotoImage(imagePIL)
    pic = canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.itemconfig(pic, image[0])
    plt.imshow(RGB_img)
    plt.show()
def geserbawah():
    global img
    image = np.array(utils.geserbawah(img=img), dtype=np.uint8)
    img = image
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagePIL = Image.fromarray(RGB_img)
    photo = ImageTk.PhotoImage(imagePIL)
    pic = canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.itemconfig(pic, image[0])
    plt.imshow(RGB_img)
    plt.show()
def histogram():
    global img
    image = utils.histogram(img)
    plt.plot(image)
    plt.show()
def histogrameq():
    global img
    image = np.array(utils.equalization(img=img), dtype=np.uint8)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.show()
def blurre():
    global img
    image = np.array(utils.blur(img=img), dtype=np.uint8)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.show()

def edgee():
    global img
    image = np.array(utils.edge(img=img), dtype=np.uint8)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.show()
def sharpen():
    global img
    image = np.array(utils.sharpen(img=img), dtype=np.uint8)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.show()
def threshold():
    global img
    image = np.array(utils.threshold(img,int(threshold_min.get()),int(threshold_max.get())), dtype=np.uint8)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.show()
def region():
    global img
    x, y, seed = int(x_grow.get()), int(y_grow.get()), int(seed_grow.get())
    print(seed)
    image = np.array(utils.growth(img,x,y, seed), dtype=np.uint8)
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.show()

topFrame = Frame(root)
topFrame.pack()

bottomFrame3 = Frame(root)
bottomFrame3.pack(side=BOTTOM)
bottomFrame2 = Frame(root)
bottomFrame2.pack(side=BOTTOM)

bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)

middleFrame = Frame(root)
middleFrame.pack(side=BOTTOM)

browse = Button(topFrame, text="Browse Image",command=change_image)
browse.pack(side=TOP)

canvas = Canvas(topFrame, height=500, width=500, background="white", bd=1, relief=tk.RAISED)
canvas.pack(fill=X)

btn_greyscale = Button(middleFrame, text="Greyscale", command=greyscale)
btn_greyscale.pack(side=LEFT)

btn_bigger = Button(middleFrame, text="Bigger", command=big)
btn_bigger.pack(side=LEFT)

btn_smaller = Button(middleFrame, text="Smaller", command=small)
btn_smaller.pack(side=LEFT)

btn_brighter = Button(middleFrame, text="Brighter", command=brighter)
btn_brighter.pack(side=LEFT)

btn_darker = Button(middleFrame, text="Darker", command=darker)
btn_darker.pack(side=LEFT)

btn_blure = Button(middleFrame, text="Blooor", command=blurre)
btn_blure.pack(side=LEFT)

btn_blure1 = Button(middleFrame, text="Edge", command=edgee)
btn_blure1.pack(side=LEFT)

btn_blure2 = Button(middleFrame, text="Sharpen", command=sharpen)
btn_blure2.pack(side=LEFT)

btn_histogram = Button(middleFrame, text="Histogreum", command=histogram)
btn_histogram.pack(side=LEFT)

btn_equalization = Button(middleFrame, text="EQ", command=histogrameq)
btn_equalization.pack(side=LEFT)


btn_dragleft = Button(bottomFrame, text="Left", command=geserkiri)
btn_dragleft.pack(side=LEFT)
btn_dragright = Button(bottomFrame, text="Right", command=geserkanan)
btn_dragright.pack(side=LEFT)
btn_dragup = Button(bottomFrame, text="Up", command=geseratas)
btn_dragup.pack(side=LEFT)
btn_dragdown = Button(bottomFrame, text="Down", command=geserbawah)
btn_dragdown.pack(side=LEFT)

label_threshold = Label(bottomFrame2,text="Treshold Segmentation :")
label_threshold.pack(side=LEFT)
threshold_min = Entry(bottomFrame2,width=5)
threshold_min.pack(side=LEFT)
threshold_max = Entry(bottomFrame2,width=5)
threshold_max.pack(side=LEFT)
btn_threshold = Button(bottomFrame2, text="GO", command=threshold)
btn_threshold.pack(side=LEFT)

label_grow = Label(bottomFrame3,text="Region Growing :")
label_grow.pack(side=LEFT)
x_grow = Entry(bottomFrame3,width=5)
x_grow.pack(side=LEFT)
y_grow = Entry(bottomFrame3,width=5)
y_grow.pack(side=LEFT)
seed_grow = Entry(bottomFrame3,width=5)
seed_grow.pack(side=LEFT)
btn_grow = Button(bottomFrame3, text="GO", command=region)
btn_grow.pack(side=LEFT)

root.mainloop()