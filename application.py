from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox, filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from Pop_up import Pop_up

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        #general features
        self.output_path = Path(__file__).parent
        self.assets_path = self.output_path / "C:/Users/zazad/OneDrive/Documents/INSA_4A/devLog/build/assets/frame2"
        self.geometry("925x722")
        self.configure(bg="#100F0F")
        self.title("Face Generator - v.1.0.0")

        #Attributes
        self.images = []
        self.selected_im = []
        self.board_button = []
        self.history_button = []
        self.click = False
        self.allow_selection = False
        self.open_button = Button(self, command=self.button_open, height = 35, width = 35)

        self.canvas = tk.Canvas(
            self,
            bg="#100F0F",
            height=722,
            width=925,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        #Lauche functions
        self.background()
        self.create_menu_bar()
        self.add_button()
        #self.scrolling_board()
        


    def background(self):
        """ Creates the background of the window """

        self.canvas.create_rectangle(
            2.0,
            11.0,
            51.0,
            747.0,
            fill="#201836",
            outline="")

        self.canvas.create_rectangle(
            57.0,
            11.0,
            788.0,
            713.0,
            fill="#201F1F",
            outline=""
            )

        self.canvas.create_rectangle(
            794.0,
            11.0,
            925.0,
            765.0,
            fill="#201F1F",
            outline="")

        self.canvas.create_text(
            830.0,
            19.0,
            anchor="nw",
            text="History",
            fill="#FFFFFF",
            font=("Inter", 16 * -1)
        )

        self.canvas.create_rectangle(
            792.0,
            44.0,
            925.0,
            46.0,
            fill="#31215F",
            outline="")

        self.canvas.create_rectangle(
            5.0,
            15.0,
            45.0,
            52.60757827758789,
            fill="#000000",
            outline="")

        self.canvas.create_rectangle(
            5.0,
            61.0,
            45.0,
            98.60757827758789,
            fill="#000000",
            outline="")

        self.canvas.create_rectangle(
            5.0,
            107.0,
            45.0,
            144.6075782775879,
            fill="#000000",
            outline="")

        self.canvas.create_rectangle(
            5.0,
            153.0,
            45.0,
            190.6075782775879,
            fill="#000000",
            outline="")

        self.resizable(False, False)

    def create_menu_bar(self):
        """ Creates a Menu bar with different sections : 
        File --> New, to open a new window and Exit, to quit the software
        Tools --> gives access to all the tools (open, select, generate and export)
        About --> gives information about the software
        """
        menu_bar = Menu(self)

        menu_file = Menu(menu_bar, tearoff=0)
        menu_file.add_command(label="New", command=self.new_window)
        menu_file.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=menu_file)

        menu_edit = Menu(menu_bar, tearoff=0)
        menu_edit.add_command(label="Select", command=self.button_select)
        menu_edit.add_separator()
        menu_edit.add_command(label="Generate", command=self.button_arrow)
        menu_edit.add_command(label="Open", command=self.button_open)
        menu_edit.add_command(label="Export", command=self.button_export)
        menu_bar.add_cascade(label="Tools", menu=menu_edit)

        menu_help = Menu(menu_bar, tearoff=0)
        menu_help.add_command(label="About", command=self.do_about)
        menu_bar.add_cascade(label="Help", menu=menu_help)

        self.config(menu=menu_bar)

    def new_window(self):
        """ This function is launched when the user clickes on  new in the menu bar
        It opens a new application window"""

        new_window = Application()

    def do_about(self):
        """ This function is launched when the user clickes on About in the menu bar
        It opens a pop up window with informations on the software"""

        messagebox.showinfo("My title", "My message")

    def changeOnHover(button, colorOnHover, colorOnLeave):
        """ Alllows to change the color of a button when we hover over it"""
 
    # adjusting backgroung of the widget
    # background on entering widget
        button.bind("<Enter>", func=lambda e: button.config(
            background=colorOnHover))
 
    # background color on leving widget
        button.bind("<Leave>", func=lambda e: button.config(
            background=colorOnLeave))

    def button_open(self):
        """ It is launched when the user clicks on the button open of on the option open in the menu bar
        It opens a pop up window called Pop_up to allow the user to select categories to narrow the choices of images to open and the number of images
        to open
        """
        self.open_button.config(state=tk.DISABLED)
        pop_up = Pop_up(self)
    
    def on_popUp_finish(self, info):
        """ This function waits for the actions in open_button and by extension Pop_up to finish before displaying the images on the board
        """
        self.open_button.config(state=tk.NORMAL)
        self.image_grid(int(info[3]))

    def button_select(self):
        """ When the button sleect ou the option select in the menu bar is clicked,
        this functions whanges the statue of allow_selection to continue operations in other functions"""
        if self.allow_selection :
            self.allow_selection = False
        else :
            self.allow_selection = True
        return self.allow_selection

    def button_arrow(self):
        """launches the genetic algorithm
        opens an error dialog if no images selected"""
        if self.selected_im :
            self.history((805, 55))
            self.clear_board()
        else :
            messagebox.showinfo("Error", "No images were selected")


    def button_export(self):
        """Opens file dialog to select folder + give name of the file (format PNG).
        Opens an error dialog if no images selected."""

        if not self.selected_im:
            # If no images are selected, show an error dialog
            messagebox.showerror("Error", "No images selected.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save File",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )

        if file_path:
            # If a file path is chosen, export the selected images to a PNG file
            self.save_image(file_path)


    """ def save_image(self, file_path):
        for i, photo_image in enumerate(self.selected_im):
            # Convert PhotoImage to PIL Image
            pil_image = self.photo_image_to_pil(photo_image)
            if pil_image:
                # Constructing a unique file name for each image
                filename = f"{file_path}/image_{i}.png"
                # Saving the image
                pil_image.save(filename)

    def photo_image_to_pil(self, photo_image):
        # Get the width and height of the PhotoImage
        width = photo_image.width()
        height = photo_image.height()
        # Create a PIL Image from the pixel data of the PhotoImage
        pil_image = Image.new("RGB", (width, height))
        # Get the pixel data from the PhotoImage
        data = photo_image.tk.call(photo_image, 'data')
        # Paste the pixel data into the PIL Image
        pil_image.frombytes(data)
        return pil_image """
        
    def open_image(self, path, dimension):
        """ Allows to open images in a given path and with given dimensions"""
        self.image = Image.open(path)
        self.resized = self.image.resize(dimension)
        self.tk_image = ImageTk.PhotoImage(self.resized)

        return self.tk_image


    def add_button(self):
        """ This function creates all the buttons in the tool bar, on the left side of the board"""
        dim = (35,35)
        self.image_open = self.open_image("images/open2.png", dim)
        self.image_select = self.open_image("images/select2.png", dim)
        self.image_round = self.open_image("images/round.png", dim)
        self.image_export = self.open_image("images/export_87484.png", dim)

        # Create a button with an image
        button_open = Button(self, image=self.image_open, command=self.button_open, height = 35, width = 35)
        button_open.place(x=5, y=15)

        button_select = Button(self, image=self.image_select, command=self.button_select, height =35, width = 35)
        button_select.place(x=5, y=61)

        button_round = Button(self, image=self.image_round, command=self.button_arrow, height =35, width = 35)
        button_round.place(x=5, y=107)

        button_export = Button(self, image=self.image_export, command=self.button_export, height =35, width = 35)
        button_export.place(x=5, y=153)

    def display_image(self, myImages, start_position, spacing):
        """ this function displays images in line at a given starting position and with a given spacing in between"""
        x_pos, y_pos = start_position
        dim = (200,200)
        for image in myImages:
            button = self.create_button(image, [x_pos, y_pos], dim)  # Create button for each image
            self.board_button.append(button)  # Append the button to the list
            x_pos += image.width() + spacing
        return self.board_button  # Return the list of buttons

    def image_grid(self, nb):
        """ Here the images are displayed in 3 by 3 grid """
        dim = (256, 256)
        padding = 0
        images_per_row = 3

        # Load images
        for i in range(nb):
            self.images.append(self.open_image("images/exemple.png", dim))

        x_pos, y_pos = (65, 20)

        for i in range(0, nb, images_per_row):
            row_images = self.images[i:i + images_per_row]
            self.display_image(row_images, (x_pos, y_pos), padding)
            y_pos += dim[1] + padding

    def clear_board(self):
        while self.board_button:
            button = self.board_button.pop()  # Remove the last button
            button.destroy()
        

    def scrolling_board(self):
        """ This function creates a canva with a scrolling bar
        TO FINISH """
        # Create the canvas
        canvas = Canvas(self, bg="#201F1F", width=720, height=713)
        canvas.place(x=57, y=11)

        # Create the vertical scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.place(x=762, y=11, height = 713)

        # Configure canvas to use the scrollbar
        canvas.config(yscrollcommand=scrollbar.set)

        # Call a method to add images to the canvas
        images = self.image_grid(5)

        #self.canvas.lower(scrollbar)

    def create_button(self, im, position, dim):
        """ In this function we can create buttons with given images on top of it, at a specific position and dimensions"""
        button = Button(self, image=im, command=lambda : self.clicked(im, button), height = dim[0], width = dim[1])
        button.place(x=position[0], y=position[1])

        return button


    def clicked(self, im, button):
        """ This function changes the appearance of the image buttons on the board when they are selected, but only if the tool selection was launched"""
        if self.allow_selection :
            if self.click and im in self.selected_im : 
                button.config(bg="SystemButtonFace")
                self.selected_im.remove(im)
                self.click = False
            else :
                button.config(bg="red")
                self.selected_im.append(im)
                self.click = True

        return self.selected_im
    
    def history(self, position):
        """ This functions creates a history on the right side of the board where all the selected images can be saved"""
        spacing = 10
        x, y = position
        dim = (100,100)
        for image in self.selected_im :
            button = self.create_button(image, [x, y], dim)
            self.history_button.append(button)
            y += 100 + spacing
        return self.history_button
    


        

if __name__ == "__main__":
    app = Application()
    app.mainloop()
