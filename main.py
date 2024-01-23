import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from keras.models import load_model
from keras.preprocessing import image as keras_preprocessing
import numpy as np
import cv2

print('Loading GUI...')

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
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5') or f.endswith('.keras')]
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

            if self.selectedModel.get().find('OCR') != -1:
                self.loaded_image = cv2.imread(filepath)
                self.display_image = cv2.resize(self.loaded_image, (250, 250))
            else:
                self.loaded_image = Image.open(filepath)
                self.display_image = self.loaded_image.resize((250, 250), Image.Resampling.LANCZOS)
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

        if self.selectedModel.get().find("OCR") == -1:
            processed_image = self.preprocess_image(self.loaded_image)

            # Run the model
            results = self.model.predict(processed_image)

        else:
            print("Runing OCR model")
            results = self.process_OCR(self.loaded_image)

        # Process the results (this will depend on your model's output)
        self.display_results(results)

# ovo je preprocesiranje slike u format koji odgovara modelu i normalizacija slike
    def preprocess_image(self, image):
        # Convert PIL image to an OpenCV image
        image = np.array(image)
        if self.model.input_shape[3] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize the image using OpenCV
        image = cv2.resize(image, (self.model.input_shape[1], self.model.input_shape[2]))

        # Convert back to RGB format if your model expects RGB inputs
        if self.model.input_shape[3] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize the image
        image = image / 255.0

        # Convert the image to numpy array and add batch dimension
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        return image_array


#rezanje slike prilikom segmentacije zbog izvlacenja konture
    def min_value_OCR(self, position):
        if position - 300 > 0:
            return position - 300

        return 0


    def max_value_OCR(self, position, border):
        if position + 300 < border:
            return position + 300

        return border

#procesiranje slike za OCR
    def process_OCR(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh_image = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY_INV)[1]

        contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

        results = [0, 0, 0, 0]
            
        for index in range(len(sorted_contours)):
            contour_moments = cv2.moments(sorted_contours[index])

            if contour_moments["m00"] < 2000.0:
                break

            cx = int(contour_moments["m10"] / contour_moments["m00"])
            cy = int(contour_moments["m01"] / contour_moments["m00"])
            
            mask = np.zeros(thresh_image.shape, np.uint8)
            cv2.drawContours(mask, sorted_contours, index, (255, 255, 255), -1)

            newW = min(cx - self.min_value_OCR(cx), self.max_value_OCR(cx, 640) - cx)
            newH = min(cy - self.min_value_OCR(cy), self.max_value_OCR(cy, 480) - cy)
            mask = mask[cy - newH:cy + newH, cx - newW:cx + newW]
            mask = cv2.resize(mask, (64, 64))
            #showImage(mask, 5)
            #cv.imwrite(adress, mask_copy_2)
            test_image = keras_preprocessing.img_to_array(mask)
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = np.vstack([test_image])
            results = self.model.predict(test_image/255.0)
            print("OCR: ", results)

            if self.process_results_OCR(results) in ["H", "S", "U"]:
                return results

        return results


# ovo je procesiranje rezultata za cifar10, treba dodati funkciju za bilo koji drugi dataset
    def process_results_cifar10(self, results):
        # Specific processing for model1
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        index = -1
        weight_table = results[0]
        n = len(weight_table)
        for i in range(0, n):
            if weight_table[i] == np.max(weight_table):
                index = i
                break
        return classes[index]


    
# procesiranje rezultata za skraćeni OCR
    def process_results_OCR(self, results):
        classes = ["H", "NONE", "S", "U"]
        result = "NOT CLASSIFIED"

        for i in range(4):
            if results[0][i] > 0.7:
                result = classes[i]

        return result


# ovdje treba pozivati posebne funkcije za razlicite modele
    def display_results(self, results):
        global result
        model_name = self.selectedModel.get()

        if model_name.find("mnist") != -1:
            print(results)
            result = np.argmax(results)
        if model_name.find('cifar10') != -1:
            print(results)
            result = self.process_results_cifar10(results)
        if model_name.find("OCR") != -1:
            print(results)
            result = self.process_results_OCR(results)
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

        load_button_width = 100
        self.loadModelButton = tk.Button(self.frame, text="Load Model", command=self.load_model)
        self.loadModelButton.place(x=20 + menu_width + 20, y=18, width=load_button_width)

        # Choosing the picture
        self.browse_button = tk.Button(self.frame, text="Browse", command=self.load_image)
        self.browse_button.place(x=20 + menu_width + 20, y=60, width=load_button_width)

        # Set an appropriate width for the Entry widget
        entry_width = 300
        self.filepath_entry = tk.Entry(self.frame, width=entry_width)
        self.filepath_entry.place(x=60 + 80, y=100)  # Adjust the x coordinate as needed

        # Run Model Button
        self.run_button = tk.Button(self.frame, text="Run Model", command=self.run_model)
        self.run_button.place(x=300, y=500, width=200, height=30)


gui = appGUI()
