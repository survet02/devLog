import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
from tkinter import *
from pathlib import Path

class Pop_up(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.output_path = Path(__file__).parent
        self.assets_path = self.output_path / "C:/Users/zazad/OneDrive/Documents/INSA_4A/build/assets/frame1"

        self.geometry("305x338")
        self.configure(bg="#0F0F0F")
        self.title("Open images")

        self.sel_gender = tk.StringVar()
        self.sel_hair = tk.StringVar()
        self.sel_skin = tk.StringVar()
        self.sel_number = tk.StringVar()

        self.selection = []


        self.create_widgets()
        self.create_options()
        self.button()

    def create_widgets(self):
        self.canvas = tk.Canvas(
            self,
            bg="#0F0F0F",
            height=338,
            width=305,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        self.create_text(14.0, 20.0, "Select gender :")
        self.create_text(14.0, 90.0, "Select hair color :")
        self.create_text(14.0, 160.0, "Select skin tone :")
        self.create_text(14.0, 230.0, "Number of images to display")
   
        self.resizable(False, False)

    def create_text(self, x, y, text):
        self.canvas.create_text(
            x,
            y,
            anchor="nw",
            text=text,
            fill="#FFFFFF",
            font=("Inter", 12)
        )


    def create_options(self):
        self.selection = []
        genders = ["Male", "Female"]
        gender = ttk.Combobox(self, values=genders, width=15, textvariable=self.sel_gender)
        gender.pack(pady=45, padx=1)
        gender['state'] = 'readonly'

        hairs = ["Blond", "Brown"]
        hair = ttk.Combobox(self, values=hairs, width=15, textvariable=self.sel_hair)
        hair.pack(pady=10, padx=1)
        hair['state'] = 'readonly'
        
        skins = ["Pale", "Dark"]
        skin = ttk.Combobox(self, values=skins, width=15, textvariable=self.sel_skin)
        skin.pack(pady=30, padx=1)
        skin['state'] = 'readonly'

        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        number = ttk.Combobox(self, values=numbers, width=15, textvariable=self.sel_number)
        number.pack(pady=35, padx=1)
        number['state'] = 'readonly'

        return [self.sel_gender, self.sel_hair, self.sel_skin, self.sel_number]
    
    def setSelection(self, box):
        for i in range(len(box)):
            self.selection.append(box[i].get())

        info = self.selection
        self.destroy()
        self.parent.on_popUp_finish(info)
        
    def button(self):
        button = Button(self, text="SAVE", command=lambda: self.setSelection(self.create_options()), height = 1, width = 7)
        button.place(x=230, y=300)



if __name__ == "__main__":
    app = Pop_up()
    app.mainloop()
