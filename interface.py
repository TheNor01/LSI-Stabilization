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



def entryClick(event):
    event.widget.delete(0,END)
    root.filename = filedialog.askopenfilename(initialdir="./gallery/",filetypes=(("mp4 files","*.mp4"),("all files","*.*")))
    event.widget.insert(0,root.filename)
    

path = StringVar()
e = Entry(root,width=65,textvariable = path)
e.bind("<Button-1>", entryClick)
e.place(x=150,y=150)
e.insert(0,"Insert Video's Path")


def StartAlgoCustom(pathVar,name,corners,blockSize):
    root.destroy()
    Stabilization1(pathVar,corners,blockSize)

def StartAlgoStab(pathVar,name,option,boolObject):
    root.destroy()
    Stabilization2(pathVar,option,boolObject)

def ChooseParamsCustom(pathVar,name):
    
    corners = IntVar()
    corners.set(100)
    labelCornes = Label(top,text="Corners",font=6).place(x=140,y=120)
    slider = Scale(top,from_=0,to=200,orient='horizontal',variable=corners)
    slider.place(x=120,y=80)

    blockSize = IntVar()
    blockSize.set(2)
    labelSize = Label(top,text="BlockSize",font=6).place(x=260,y=120)
    slider2 = Scale(top,from_=1,to=15,orient='horizontal',variable=blockSize)
    slider2.place(x=250,y=80)
    confirm = Button(top,text="Confirm your option",command=lambda: StartAlgoCustom(pathVar,name,slider.get(),slider2.get()),state="normal",font=18)
    confirm.place(x=240,y=320)

def ChooseAlgoStab(pathVar,name):
    
    clicked = StringVar()
    clicked.set("FAST")
    featureOption = Label(top,text="Feature Tracking Option",font=24).place(x=106,y=184)

    drop = OptionMenu(top,clicked,"FAST","BRISK","ORB","GFTT","HARRIS","DENSE")
    drop.place(x=106,y=204)
    chkValue = BooleanVar() 
    chkValue.set(False)
    checkBoxObject= Checkbutton(top, text='Use Object Detector', var=chkValue)
    checkBoxObject.place(x=106,y=250)
    confirm = Button(top,text="Confirm your option",command=lambda: StartAlgoStab(pathVar,name,clicked.get( ),chkValue.get()),state="normal",font=18)
    confirm.place(x=240,y=320)
        
    

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
    top.minsize(400,400)
    top.title("Choose your algorithm")
    pathLabel = Label(top,text="PATH:   "+pathVar).place(x=0,y=0)
    buttonCustomAlgo = Button(top,text="Custom",command=lambda: ChooseParamsCustom(pathVar,"Custom"),font=6,height = 8, width = 10).place(x=0,y=40)
    buttonVidStabAlgo = Button(top,text="VidStab",command=lambda: ChooseAlgoStab(pathVar,"VidStab"),font=6,height = 8, width = 10).place(x=0,y=180)
    



infoProject = Label(root,text="L.S.I Let's stabilize it",font=14)
infoProject.place(x=280,y=60)

infoAboutMe = Label(root,text="Aldo Fiorito Multimedia LM-18",font=14)
infoAboutMe.place(x=240,y=390)

start = Button(root,text="Start",command=StartUp,state="normal",font=18)
start.place(x=320,y=250)

#infoAboutMe.grid(row=1,column=0)
#infoProject.grid(row=0,column=0)


root.mainloop()