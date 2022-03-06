from tkinter import *
from tkinter import filedialog
from main import Stabilization1,Stabilization2
from tkinter import messagebox

root = Tk()
root.geometry("700x500")
root.resizable(False,False)
root.title("Image stabilization")
root.config(background="#756d7d")
root.canvas=Canvas(width=400,height=350)
root.canvas.place(x=150,y=60)

global clicked
clicked = StringVar()
clicked.set("None")

global corners
corners = IntVar()
corners.set(1)

global checkStab
checkStab = 0
global checkCustom
checkCustom = 0

def entryClick(event):
    event.widget.delete(0,END)
    root.filename = filedialog.askopenfilename(initialdir="./gallery/",filetypes=(("mp4 files","*.mp4"),("all files","*.*")))
    event.widget.insert(0,root.filename)
    

path = StringVar()
e = Entry(root,width=65,textvariable = path)
e.bind("<Button-1>", entryClick)
e.place(x=150,y=150)
e.insert(0,"Insert Video's Path")


def StartAlgoCustom(pathVar,corners,blockSize):
    root.destroy()
    Stabilization1(pathVar,corners,blockSize)

def StartAlgoStab(pathVar,option,boolObject,thresHold):
    print("startAlgo")
    if(option == "None"):
        messagebox.showerror("Error", "Choose algoritm")
        return
    root.destroy()
    Stabilization2(pathVar,option,boolObject,thresHold)

def ChooseParamsCustom(pathVar,name):
    print("custom")
    global checkCustom
    checkCustom=1
    print("check custom",checkCustom)
    print("check stab",checkStab)
    print(clicked.get())


    if checkStab==1:
        featureOption.destroy()
        drop.destroy()
        treshLabel.destroy()
        sliderTreshHold.destroy()
        checkBoxObject.destroy()

    global labelCornes
    labelCornes = Label(top,text="Corners",font=6)
    labelCornes.place(x=140,y=120)

    global slider
    slider = Scale(top,from_=0,to=200,orient='horizontal',variable=corners)
    slider.place(x=120,y=80)

    blockSize = IntVar()
    blockSize.set(2)

    global labelSize
    labelSize = Label(top,text="BlockSize",font=6)
    labelSize.place(x=260,y=120)

    global slider2
    slider2 = Scale(top,from_=1,to=15,orient='horizontal',variable=blockSize)
    slider2.place(x=250,y=80)


    confirm = Button(top,text="Confirm your option",command=lambda: StartAlgoCustom(pathVar,slider.get(),slider2.get()),state="normal",font=18)
    confirm.place(x=240,y=360)


def ChooseAlgoStab(pathVar,name):
    print("stab")
    global checkStab
    checkStab = 1
    print("check custom",checkCustom)
    print("check stab",checkStab)

    if corners.get() != 1 or checkCustom==1:
        labelCornes.destroy()
        slider.destroy()
        labelSize.destroy()
        slider2.destroy()

    global featureOption
    featureOption = Label(top,text="Feature Tracking Option",font=14)
    featureOption.place(x=106,y=184)

    global drop
    drop = OptionMenu(top,clicked,"FAST","BRISK","ORB","GFTT","HARRIS","DENSE")
    drop.place(x=300,y=184)
    chkValue = BooleanVar() 
    chkValue.set(False)

    thresHold=IntVar()
    thresHold.set(5)
    global treshLabel
    treshLabel = Label(top,text="Treshold, only for FAST")
    treshLabel.place(x=106,y=220)
    
    global sliderTreshHold
    sliderTreshHold=Scale(top,from_=5,to=40,orient='horizontal',length=150,tickinterval=5,variable=thresHold)
    sliderTreshHold.place(x=106,y=250)

    global checkBoxObject
    checkBoxObject= Checkbutton(top, text='Use Object Detector', var=chkValue)
    checkBoxObject.place(x=240,y=320)

    confirmStab = Button(top,text="Confirm your option",command=lambda: StartAlgoStab(pathVar,clicked.get(),chkValue.get(),sliderTreshHold.get()),state="normal",font=15)
    confirmStab.place(x=240,y=360)
    
    

def StartUp():
    global pathVar
    pathVar = path.get()
    if(pathVar=="Insert Video's Path"):
        print("Campi non validi")
        messagebox.showerror("Error", "Scegliere path")
        return
    global top
    global confirm
    confirm = Button()
    top = Toplevel()
    top.config(background="#756d7d")
    top.minsize(400,400)
    top.title("Choose your algorithm")
    pathLabel = Label(top,text="PATH:   "+pathVar)
    pathLabel.place(x=0,y=0)
    buttonCustomAlgo = Button(top,text="Custom",command=lambda: ChooseParamsCustom(pathVar,"Custom"),font=6,height = 8, width = 10)
    buttonCustomAlgo.place(x=0,y=40)
    buttonVidStabAlgo = Button(top,text="VidStab",command=lambda: ChooseAlgoStab(pathVar,"VidStab"),font=6,height = 8, width = 10)
    buttonVidStabAlgo.place(x=0,y=180)
    



infoProject = Label(root,text="L.S.I",font=14)
infoProject.place(x=330,y=60)
infoProject2 = Label(root,text="Let's stabilize it",font=14)
infoProject2.place(x=290,y=90)

infoAboutMe = Label(root,text="Aldo Fiorito Multimedia LM-18",font=14)
infoAboutMe.place(x=240,y=390)

start = Button(root,text="Start",command=StartUp,state="normal",font=18)
start.place(x=320,y=250)

#infoAboutMe.grid(row=1,column=0)
#infoProject.grid(row=0,column=0)


root.mainloop()   