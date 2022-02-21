from tkinter import *
from tkinter import filedialog


root = Tk()
root.minsize(200,200)
root.title("Image stabilization")

def entryClick(event):
    event.widget.delete(0,END)
    root.filename = filedialog.askopenfilename(initialdir="./gallery/",filetypes=(("mp4 files","*.mp4"),("all files","*.*")))
    event.widget.insert(0,root.filename)
    
    

path = StringVar()
e = Entry(root,width=100,textvariable = path)
e.bind("<Button-1>", entryClick)
e.grid(row=1,column=0,columnspan=3,pady=10)
e.insert(0,"Ïnsert Video's Path")

def StartUp():
    global pathVar
    pathVar = path.get()
    top = Toplevel()
    top.minsize(200,200)
    top.title("Choose your algorithm")
    pathLabel = Label(top,text="PATH:   "+pathVar).grid(row=0,columnspan=2)
    buttonCustomAlgo = Button(top,text="Custom").grid(row=1,columnspan=4)
    buttonVidStabAlgo = Button(top,text="VidStab").grid(row=2,columnspan=4)
    



infoProject = Label(root,text="L.S.I Let's stabilize it")
infoProject.grid(row=0,column=1)

infoAboutMe = Label(root,text="Aldo Fiorito Multimedia LM-18")
infoAboutMe.grid(row=5,column=0, columnspan=3)

start = Button(root,text="Run!",command=StartUp,state="normal")
start.grid(row=2,column=1)

#infoAboutMe.grid(row=1,column=0)
#infoProject.grid(row=0,column=0)


root.mainloop()