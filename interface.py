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


def StartAlgo(pathVar,name):
    root.destroy()
    Main(pathVar,name)
    
    

def StartUp():
    
    global pathVar
    pathVar = path.get()
    if(pathVar=="Insert Video's Path"):
        print("Campi non validi")
        messagebox.showerror("Error", "Scegliere path")
        return
    top = Toplevel()
    top.minsize(200,200)
    top.title("Choose your algorithm")
    pathLabel = Label(top,text="PATH:   "+pathVar).grid(row=0,columnspan=2)
    buttonCustomAlgo = Button(top,text="Custom",command=lambda: StartAlgo(pathVar,"Custom")).grid(row=1,columnspan=4)
    buttonVidStabAlgo = Button(top,text="VidStab",command=lambda: StartAlgo(pathVar,"VidStab")).grid(row=2,columnspan=4)
    



infoProject = Label(root,text="L.S.I Let's stabilize it",font=14)
infoProject.place(x=280,y=60)

infoAboutMe = Label(root,text="Aldo Fiorito Multimedia LM-18",font=14)
infoAboutMe.place(x=240,y=390)

start = Button(root,text="Run!",command=StartUp,state="normal",font=18)
start.place(x=320,y=250)

#infoAboutMe.grid(row=1,column=0)
#infoProject.grid(row=0,column=0)


root.mainloop()