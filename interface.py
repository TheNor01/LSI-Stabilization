from tkinter import *
from tkinter import filedialog
from main import Main
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


def StartAlgoCustom(pathVar,name):
    root.destroy()
    Main(pathVar,name)

def StartAlgoStab(pathVar,name,option):
    root.destroy()
    Main(pathVar,name,option)

def ChooseAlgoStab(pathVar,name):
    clicked = StringVar()
    clicked.set("FAST")
    featureOption = Label(top,text="Feature Tracking Option",font=24).grid(row=5,columnspan=4)

    drop = OptionMenu(top,clicked,"FAST","BRISK","ORB","GFTT","HARRIS","DENSE")
    drop.grid(row=6,columnspan=4)
    confirm = Button(top,text="Confirm your option",command=lambda: StartAlgoStab(pathVar,name,clicked.get()),state="normal",font=18)
    confirm.grid(row=8,columnspan=4)
        
    

def StartUp():
    global pathVar
    pathVar = path.get()
    if(pathVar=="Insert Video's Path"):
        print("Campi non validi")
        messagebox.showerror("Error", "Scegliere path")
        return
    global top
    top = Toplevel()
    top.minsize(200,200)
    top.title("Choose your algorithm")
    pathLabel = Label(top,text="PATH:   "+pathVar).grid(row=0,columnspan=2)
    buttonCustomAlgo = Button(top,text="Custom",command=lambda: StartAlgoCustom(pathVar,"Custom"),font=8).grid(row=3,columnspan=4)
    buttonVidStabAlgo = Button(top,text="VidStab",command=lambda: ChooseAlgoStab(pathVar,"VidStab"),font=8).grid(row=4,columnspan=4)
    



infoProject = Label(root,text="L.S.I Let's stabilize it",font=14)
infoProject.place(x=280,y=60)

infoAboutMe = Label(root,text="Aldo Fiorito Multimedia LM-18",font=14)
infoAboutMe.place(x=240,y=390)

start = Button(root,text="Start",command=StartUp,state="normal",font=18)
start.place(x=320,y=250)

#infoAboutMe.grid(row=1,column=0)
#infoProject.grid(row=0,column=0)


root.mainloop()