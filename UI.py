import tkinter
from PIL import Image,ImageDraw
from Inference import inference

class MyCanvas:
    def __init__(self,root):
        self.root=root
        self.canvas=tkinter.Canvas(root,width=256,height=256,bg='black')
        self.canvas.pack()
        self.image1 = Image.new("RGB", (256, 256), "black")
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind('<B1-Motion>',self.Draw)


    def Draw(self,event):
        self.canvas.create_oval(event.x,event.y,event.x,event.y,outline="white",width = 20)
        self.draw.ellipse((event.x-10,event.y-10,event.x+10,event.y+10),fill=(255,255,255))

root=tkinter.Tk()
root.geometry('300x300')
frame=tkinter.Frame(root,width=256,height=256)
frame.pack_propagate(0)
frame.pack(side='top')
canvas1=MyCanvas(frame)

def inference_click():
    img = canvas1.image1
    print(inference(img))
    canvas1.canvas.delete("all")
    canvas1.image1 = Image.new("RGB", (256, 256), "black")
    canvas1.draw = ImageDraw.Draw(canvas1.image1)
    

btn_Inference=tkinter.Button(root,
            text="Inference",         
            width=7,           
            height=1,                
            command=inference_click      
            )
btn_Inference.pack()

root.mainloop()
