import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from keras.models import load_model
import numpy as np
import cv2

class appGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("800x600")
        self.root.title("ModelTester")
        self.model_files = self.get_model_names()
        self.create_widgets()
        self.root.mainloop()

# za dohvacanje imena modela
    def get_model_names(self):
        model_dir = 'models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        return model_files

    def load_model(self):
        try:
            model_path = os.path.join('models', self.selectedModel.get())
            self.model = load_model(model_path)
            messagebox.showinfo("Model Loaded", f"Model '{self.selectedModel.get()}' successfully loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the model: {e}")

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if filepath:
            self.filepath_entry.delete(0, tk.END)  # Clear the existing text
            self.filepath_entry.insert(0, filepath)  # Insert the new filepath
            self.loaded_image = Image.open(filepath)
            self.display_image = self.loaded_image.resize((250, 250), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(self.display_image)
            messagebox.showinfo("Image Loaded", "Image loaded successfully!")


    def run_model(self):
        # Check if the model is loaded
        try:
            if self.model:
                print('Model loaded')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the model: {e}")

        # Check if an image is loaded
        if not hasattr(self, 'loaded_image') or self.loaded_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return  # Exit the function if no image is loaded

        processed_image = self.preprocess_image(self.loaded_image)

        # Run the model
        results = self.model.predict(processed_image)

        # Process the results (this will depend on your model's output)
        self.display_results(results)

# ovo je preprocesiranje slike u 32x32 i normalizacija slike, ovo je za cifar10
    def preprocess_image(self, image):
        # Convert PIL image to an OpenCV image
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize the image using OpenCV
        image = cv2.resize(image, (32, 32))

        # Convert back to RGB format if your model expects RGB inputs
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize the image
        image = image / 255.0

        # Convert the image to numpy array and add batch dimension
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        return image_array

# ovo je procesiranje rezultata za cifar10, treba dodati funkciju za bilo koji drugi dataset
    def process_results_cifar10(self, results):
        # Specific processing for model1
        global max
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        index = -1
        weight_table = results[0]
        n = len(weight_table)
        max = np.max(weight_table)
        for i in range(0, n):
            if weight_table[i] == max:
                index = i
                break
        return classes[index]

# ovdje treba pozivati posebne funkcije za razlicite modele
    def display_results(self, results):
        global result
        model_name = self.selectedModel.get()
        if model_name == 'cifar10.h5':
            print(results)
            result = self.process_results_cifar10(results)
        messagebox.showinfo("Results", "The picture displays " + str(result))

    def create_widgets(self):

        self.root.resizable(False, False)  # Disable window resizing

        self.frame = tk.Frame(self.root, bg="gray")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Choosing the model
        if not self.model_files:
            self.model_files = ['No models available']

        self.selectedModel = tk.StringVar(self.frame)
        self.selectedModel.set(self.model_files[0])

        menu_width = 200
        self.modelMenu = tk.OptionMenu(self.frame, self.selectedModel, *self.model_files)
        self.modelMenu.place(x=20, y=20, width=menu_width)

        self.loadModelButton = tk.Button(self.frame, text="Load Model", command=self.load_model)
        self.loadModelButton.place(x=20 + menu_width + 20, y=18)

        # Choosing the picture
        self.browse_button = tk.Button(self.frame, text="Browse", command=self.load_image)
        self.browse_button.place(x=60, y=60)

        # Set an appropriate width for the Entry widget
        entry_width = 300
        self.filepath_entry = tk.Entry(self.frame, width=entry_width)
        self.filepath_entry.place(x=60 + 80, y=60)  # Adjust the x coordinate as needed

        # Run Model Button
        self.run_button = tk.Button(self.frame, text="Run Model", command=self.run_model)
        self.run_button.place(x=300, y=500, width=200, height=30)


gui = appGUI()
