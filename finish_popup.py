import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
from tkinter import *
from pathlib import Path

class finish_popup(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.output_path = Path(__file__).parent
        self.assets_path = self.output_path / "C:/Users/zazad/OneDrive/Documents/INSA_4A/devlog/build/assets/frame1"

        self.geometry("305x338")
        self.configure(bg="#0F0F0F")
        self.title("Final Image")

        self.background()
        self.button()
        self.image()

    def background(self):
        self.canvas = Canvas(
            self,
            bg = "#0F0F0F",
            height = 338,
            width = 305,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas.place(x = 0, y = 0)
        self.canvas.create_rectangle(
            224.021484375,
            294.3404846191406,
            290.55499267578125,
            324.00000190734863,
            fill="#000000",
            outline="")

        self.canvas.create_rectangle(
            32.0,
            59.0,
            273.0,
            280.0,
            fill="#322160",
            outline="")

        self.canvas.create_text(
            24.0,
            24.0,
            anchor="nw",
            text="Here is your final profile :",
            fill="#FFFFFF",
            font=("Inter", 16 * -1)
        )

    def button(self):
        button = Button(self, text="Restart", command=lambda: self.restart(), height = 1, width = 7)
        button.place(x=230, y=300)

    def image(self):
        image_path = self.parent.selected_im[0]
        self.label = tk.Label(self.parent, image=image_path)
        self.label.place(x=30, y=65)

    def restart(self):
        self.parent.selected_im.clear()
        self.parent.clear_board()
        self.destroy()

if __name__ == "__main__":
    app = finish_popup()
    app.mainloop()