from pathlib import Path
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox, filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from Pop_up import Pop_up
from finish_popup import finish_popup

import os
import wget
import zipfile
import torch
import torchvision.datasets as datasets
import torchvision.transforms as tfms
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import tempfile
from torchvision.transforms import ToTensor
import io
import pandas as pd

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        #general features
        self.output_path = Path(__file__).parent
        self.assets_path = self.output_path / "C:/Users/zazad/OneDrive/Documents/INSA_4A/devLog/build/assets/frame2"
        self.geometry("925x722")
        self.configure(bg="#100F0F")
        self.title("FaceGuessr - v.1.0.0")

        #Attributes
        self.images = []
        self.selected_im = []
        self.generated_image = []
        self.board_button = []
        self.history_buttons = []  # Store all history buttons
        self.click = False
        self.allow_selection = False
        self.button_bg = []
        self.open_button = Button(self, command=self.button_open, height=35, width=35)
        self.button_sel = Button(self)

        self.images_algo_gen = []
        self.possible_index = []
        self.history_initial_position = (805, 55)

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
            747.0,
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
        menu_edit.add_command(label="Export", command=self.button_finish)
        menu_bar.add_cascade(label="Tools", menu=menu_edit)

        menu_help = Menu(menu_bar, tearoff=0)
        menu_help.add_command(label="About", command=self.do_about)
        menu_bar.add_cascade(label="Help", menu=menu_help)

        self.config(menu=menu_bar)

    def new_window(self):
        """ This function clears the page to start a new simulation"""

        self.clear_board()
        self.clear_history()
        self.button_sel.configure(background = "white")

    def do_about(self):
        """Displays information about the software."""

        message = """   FaceGuessr was build by four students in 4th year at INSA Lyon in the Department of BioInformatics.
        The software aims at generating images of faces according to a genetic algorithm. The idea is to display faces from a known database and to create new face from selected ones."""


        messagebox.showinfo("About FaceGuessr", message)


    def button_open(self):
        """Opens a Pop_up window to select categories and the number of images to open."""
        self.open_button.config(state=tk.DISABLED)
        pop_up = Pop_up(self)
    
    def on_popUp_finish(self, info):
        """Waits for actions in open_button/Pop_up to finish before displaying the images on the board."""
        self.open_button.config(state=tk.NORMAL)
        gender = 0
        blond_hair = 0
        brown_hair = 0
        pale_skin = 0
        path_csv = "list_attr_test.csv"
        if info[0] == 'Male':
            gender = 1
        if info[1] == 'Blond':
            blond_hair = 1
        elif info[1] == 'Brown':
            brown_hair = 1
        if info[2] == 'Pale':
            pale_skin = 1

        self.possible_index = data_selected(test_dataset, gender, blond_hair, brown_hair, pale_skin)
        if (len(self.possible_index) >= int(info[3])):
            images,index = images_initiales(int(info[3]),self.possible_index,test_dataset)
            index = sorted(index, reverse=True)
            for i in index:
                self.possible_index.remove(self.possible_index[i])
            self.image_grid(int(info[3]),images)
            
        else:
            messagebox.showinfo("Error", ("Not enough images to show. Maximum number of images = ",len(data_index))) # type: ignore

    def button_select(self):
        """Changes the state of allow_selection to continue operations in other functions."""
        if self.allow_selection :
            self.allow_selection = False
            self.button_sel.configure(background = "white")
        else :
            self.allow_selection = True
            self.button_sel.configure(background = "#9281C1")
        return self.allow_selection

    def button_arrow(self):
        """launches the genetic algorithm
        opens an error dialog if no images selected"""
        if self.selected_im:
            dim = (256, 256)
        
        # Create history buttons for selected images
            self.history(self.history_initial_position)  # Ensure history buttons are created
        
            for i in range(len(self.selected_im)):
                if self.selected_im[i].width() != 256:
                    self.selected_im[i] = self.resize_im(self.selected_im[i], dim)
        
            converted = convert_image_to_tensor(self.selected_im)
            self.images_algo_gen = algo_gen(converted)
            self.generated_image = self.images_algo_gen
            self.images_algo_gen = [self.images_algo_gen[i].squeeze(0) for i in range(len(self.images_algo_gen))]
            
            new = len(self.images) - len(self.selected_im)
            new_images, index = images_initiales(new, self.possible_index, test_dataset)
            self.clear_board()
            index = sorted(index, reverse=True)
            for i in range(len(new_images)):
                self.possible_index.remove(self.possible_index[index[i]])
                self.images_algo_gen.append(new_images[i])

            self.image_grid(len(self.images_algo_gen), self.images_algo_gen)
            self.highlight_generated(self.generated_image)
        
            self.allow_selection = False
            self.button_sel.configure(background="white")
            self.generated_image = []
            self.selected_im = []  # Clear the list of selected images
            if self.history_buttons:
                for button in self.history_buttons:
                    button.configure(borderwidth = 0)
        else:
            messagebox.showinfo("Error", "No images were selected")

    def button_finish(self):
        """Opens a pop-up window that displays the final selected image."""

        if not self.selected_im:
            # If no images are selected, show an error dialog
            messagebox.showerror("Error", "No images selected.")
            return
        elif len(self.selected_im)>1 :
            messagebox.showerror("Error", "More than 1 image selected")
            return
        else : 
            pop_up = finish_popup(self)

        
    def open_image(self, path, dimension):
        """Opens images in a given path with given dimensions."""
        self.image = Image.open(path)
        self.resized = self.image.resize(dimension)
        self.tk_image = ImageTk.PhotoImage(self.resized)

        return self.tk_image
        
    def open_image_reconstructed(self, reconstructed_image, dimension):
        """Converts a reconstructed image tensor to a Tkinter PhotoImage and resizes it."""
        # Convert the reconstructed image tensor to a numpy array and then to a PIL Image
        np_image = reconstructed_image.detach().cpu().permute(1, 2, 0).numpy()
        self.image = Image.fromarray(np.uint8(np_image * 255))
        # Resize the image
        self.resized = self.image.resize(dimension)
        # Convert the resized image to a Tkinter PhotoImage
        self.tk_image = ImageTk.PhotoImage(self.resized)
    
        return self.tk_image


    def add_button(self):
        """Creates all the buttons in the toolbar."""
        dim = (35,35)
        self.image_open = self.open_image("images/open2.png", dim)
        self.image_select = self.open_image("images/select2.png", dim)
        self.image_round = self.open_image("images/round.png", dim)
        self.image_export = self.open_image("images/export_87484.png", dim)

        # Create a button with an image
        button_open = Button(self, image=self.image_open, command=self.button_open, height = 35, width = 35)
        button_open.place(x=5, y=15)

        self.button_sel = Button(self, image=self.image_select, command=self.button_select, height =35, width = 35)
        self.button_sel.place(x=5, y=61)

        button_round = Button(self, image=self.image_round, command=self.button_arrow, height =35, width = 35)
        button_round.place(x=5, y=107)

        button_fin = Button(self, image=self.image_export, command=self.button_finish, height =35, width = 35)
        button_fin.place(x=5, y=153)

    def display_image(self, myImages, start_position, spacing):
        """Displays images in line at a given starting position and with a given spacing."""
        x_pos, y_pos = start_position
        dim = (200,200)
        for image in myImages:
            button = self.create_button(image, [x_pos, y_pos], dim)  # Create button for each image
            self.board_button.append(button)  # Append the button to the list
            x_pos += image.width() + spacing
        return self.board_button  # Return the list of buttons

    def image_grid(self, nb, list_image):
        """Displays images in a 3 by 3 grid."""
        dim = (256, 256)
        padding = 0
        images_per_row = 3

        # Load images
        #for i in range(nb):
        #    self.images.append(self.open_image("images/exemple.png", dim))
        #self.images = [self.open_image_reconstructed(image, dim) for i in range(len(images_initiales(nb,test_dataset)))]

        if isinstance(list_image[0], PhotoImage):
            self.images = list_image
        else:
            self.images = [self.open_image_reconstructed(image, dim) for image in list_image]

        x_pos, y_pos = (65, 20)

        for i in range(0, nb, images_per_row):
            row_images = self.images[i:i + images_per_row]
            self.display_image(row_images, (x_pos, y_pos), padding)
            y_pos += dim[1] + padding
        
    def highlight_generated(self, image):
        """Highlights generated images."""
        if self.generated_image :
            for i in range(len(image)):
                self.board_button[i].configure(background = "#ff9e00", borderwidth=5)
    

    def clear_board(self):
        """Clears the board."""
        # Destroy all buttons
        for button in self.board_button:
            button.destroy()

        # Clear the lists
        self.board_button.clear()

    def clear_history(self):
        """Clears the history."""
        for button in self.history_buttons:
            button.destroy()
    
        self.history_buttons.clear()
  

    def history(self, position):
        """ This functions creates a history on the right side of the board where all the selected images can be saved"""
        spacing = 10
        x, y = position
        dim = (100, 100)
        max_history_images = 6
    
        # Create buttons for newly selected images
        new_buttons = []
        
        nb_images_tot = len(self.history_buttons) + len(self.selected_im)
        
        for image in self.selected_im:
            tk_image = self.resize_im(image, dim)
            button = self.create_button(tk_image, [x, y], dim)
            new_buttons.append(button)
            y += 100 + spacing
            if y == 715:
                y = 55
        
        self.history_initial_position = (x,y)
        # Append new buttons to the existing history_buttons list
        for i in range((nb_images_tot-max_history_images), len(self.history_buttons)):
            new_buttons.append(self.history_buttons[i])
        self.history_buttons = new_buttons
        return self.history_buttons

    
    def create_button(self, im, position, dim):
        """Creates buttons with given images."""
        button = Button(self, image=im, command=lambda : self.clicked(im, button), height = dim[0], width = dim[1], borderwidth=0)
        button.place(x=position[0], y=position[1])

        return button


    def clicked(self, im, button):
        """Changes the appearance of image buttons on the board when they are selected."""
        if self.allow_selection :
            if self.click and im in self.selected_im : 
                button.config(borderwidth=0)
                self.selected_im.remove(im)
                self.click = False
            else :
                button.config(borderwidth=5)
                button.config(background = "#9281C1")
                self.selected_im.append(im)
                self.click = True

        return self.selected_im

    def resize_im(self, im, dim):
        """Resizes images."""
        img_PIL = ImageTk.getimage(im)
        resized = img_PIL.resize(dim)
        tk_image = ImageTk.PhotoImage(resized)

        return tk_image

class ConvAE3(nn.Module): #tester 3 ->8 8->16 16->32, prendre images en 128, VV -> batch_size de 8
    def __init__(self):
        super(ConvAE3, self).__init__()
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # RVB, 256x256x3 -> 256x256x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256x256x16 -> 128x128x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128x128x16 -> 64*64*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*64*32 -> 32*32*32
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),  # 16*16*16 -> 16*16*16
            nn.ReLU(),
            #nn.Flatten()  # Aplatir les features en 1D
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16*16*16, 32*32*16)
        # Décodeur
        self.decoder = nn.Sequential(
            #nn.Linear(64*64*4, 64*64*8),  # Correction de la taille de l'entrée
            nn.ReLU(),
            nn.Unflatten(1, (16, 32, 32)),  # Correction de la taille du tensor
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32*32*16 -> 64*64*32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64*64*32 -> 128*128*16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128*128*16 -> 256x256x3
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.decoder(x)
        return x

    def encodage(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.linear(x)

    def decodage(self, x):
        x = self.decoder(x)

        
def crossover(selections):
    produits = []
    for i in range(len(selections)-1):
        crossover_point = 2600
        
        if (selections[i].shape[0] == 1):
            prod1 = torch.cat((selections[i][0][:crossover_point], selections[i+1][0][crossover_point:]), dim=0)
            prod2 = torch.cat((selections[i+1][0][:crossover_point], selections[i][0][crossover_point:]), dim=0)

        else:
            prod1 = torch.cat((selections[i][:crossover_point], selections[i+1][crossover_point:]), dim=0)
            prod2 = torch.cat((selections[i+1][:crossover_point], selections[i][crossover_point:]), dim=0)

        produits.append(prod1)
        if (i == len(selections)-2):
            produits.append(prod2)
    return produits
           
def images_initiales(nombre_images,images_possibles_index,images_tot):
    if len(images_possibles_index)>= nombre_images:
        random_index = random.sample(range(len(images_possibles_index)), nombre_images)
        images = [images_tot[images_possibles_index[index]][0].to(device) for index in random_index]
        reconstructed = [autoencoder(images[i].unsqueeze(0)).squeeze(0) for i in range(len(images))]
        return reconstructed,random_index
    else:
        return messagebox.showinfo("Error", "No more images left")
        

    
def algo_gen(selected):
    encoded = [autoencoder.flatten(autoencoder.encoder(selected[i].permute(1, 2, 0).unsqueeze(0).permute(0, 2, 1, 3))) for i in range(len(selected))]
    #encoded = [encodage(selected[i].permute(1, 2, 0).unsqueeze(0).permute(0, 2, 1, 3)) for i in range(len(selected))]
    mutated = crossover(encoded)
    decoded = [autoencoder.decoder(autoencoder.linear(mutated[i]).unsqueeze(0)) for i in range(len(mutated))]
    #decoded = [decodage(mutated[i].unsqueeze(0)) for i in range(len(mutated))]
    return decoded
    

def convert_image_to_tensor(list_images):
    converted = []
    for image in list_images:
        # Convert Tkinter PhotoImage to PIL Image
        pil_image = ImageTk.getimage(image)
        
        pil_image = pil_image.rotate(90)
        
        # Convert PIL Image to NumPy array
        numpy_array = np.array(pil_image)[:,:,:3]/256

        # Convert NumPy array to PyTorch tensor
        tensor_image = torch.tensor(numpy_array, dtype=torch.float)
        
        converted.append(tensor_image)
    return converted
    

def data_selected(images_tot, gender, blond_hair, brown_hair, pale_skin):
    # Get all targets from the dataset
    targets = images_tot.attr.clone().detach()

    # Create a mask to filter out the desired samples
    mask = (targets[:, 9] == blond_hair) & (targets[:, 11] == brown_hair) & (targets[:, 20] == gender) & (targets[:, 26] == pale_skin)
    
    # Use the mask to filter out indices of the desired samples
    possible_indices = torch.nonzero(mask).squeeze().tolist()
    
    return possible_indices
    

if __name__ == "__main__":
    data_root = "datasets"
    image_size = 256
    transforms = tfms.Compose(
        [
            tfms.Resize((image_size, image_size)),
            tfms.ToTensor(),
        ]
    )
    test_dataset = datasets.CelebA(data_root, split="test", target_type=["attr", "landmarks"], transform=transforms)
    
        
    # Créer une instance de votre autoencodeur
    autoencoder = ConvAE3()

    # Charger les poids entraînés dans votre modèle à partir du fichier
    autoencoder.load_state_dict(torch.load("autoencoder_v7.pth", map_location=torch.device('cpu')))
    # Mettre votre modèle en mode évaluation
    autoencoder.eval()

    encodeur = autoencoder.encoder
    decodeur = autoencoder.decoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder.to(device)

    app = Application()
    app.mainloop()
