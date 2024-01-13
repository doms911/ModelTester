import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from keras.models import load_model
import numpy as np

class appGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("800x600")
        self.root.title("ModelTester")
        self.model_files = self.get_model_names()
        self.create_widgets()
        self.root.mainloop()

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
            self.filepath_entry.delete(0, tk.END)
            self.filepath_entry.insert(0, filepath)
            self.loaded_image = Image.open(filepath)
            self.display_image = self.loaded_image.resize((250, 250), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(self.display_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
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


    def preprocess_image(self, image):
        # Adjust this method according to your model's requirements
        image = image.resize((32, 32))  # Example resize, adjust to your model's input size
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array

    def process_results_cifar10(self, results):
        # Specific processing for model1
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        index = -1
        weight_table = results[0]
        n = len(weight_table)
        for i in range(0, n):
            if weight_table[i] == 1:
                index = i
                break
        return classes[index]

    def display_results(self, results):
        global result
        model_name = self.selectedModel.get()
        if model_name == 'cifar10.h5':
            result = self.process_results_cifar10(results)
        messagebox.showinfo("Results", "The picture displays " + str(result))

    def create_widgets(self):
        self.frame = tk.Frame(self.root, bg="gray")
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.dropdown_frame = tk.Frame(self.frame)
        self.dropdown_frame.grid(row=0, column=0, sticky="nsew")

        self.selectLabel = tk.Label(self.dropdown_frame, text="Select a model:")
        self.selectLabel.pack()

        if not self.model_files:
            self.model_files = ['No models available']

        self.selectedModel = tk.StringVar(self.frame)
        self.selectedModel.set(self.model_files[0] if self.model_files else 'No models available')

        self.modelMenu = tk.OptionMenu(self.dropdown_frame, self.selectedModel, *self.model_files)
        self.modelMenu.pack()

        self.loadModelButton = tk.Button(self.dropdown_frame, text="Load Model", command=self.load_model)
        self.loadModelButton.pack()

        self.image_frame = tk.Frame(self.frame)
        self.image_frame.grid(row=0, column=1, sticky="nsew")

        self.browse_frame = tk.Frame(self.image_frame)
        self.browse_frame.pack(pady=10)

        self.filepath_label = tk.Label(self.browse_frame, text="Selected Image:")
        self.filepath_label.pack(side=tk.LEFT, padx=(5, 0))

        self.filepath_entry = tk.Entry(self.browse_frame, width=50)
        self.filepath_entry.pack(side=tk.LEFT, padx=5)

        self.browse_button = tk.Button(self.browse_frame, text="Browse", command=self.load_image)
        self.browse_button.pack(side=tk.LEFT, padx=(5, 0))

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        # Additional windows (window 3 and window 4) can be added here
        # Run model button
        self.run_button = tk.Button(self.dropdown_frame, text="Run Model", command=self.run_model)
        self.run_button.pack()


gui = appGUI()
