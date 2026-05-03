import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
# import joblib
from PIL import Image, ImageTk

class MacroApp(tk.Tk):
    # Desktop GUI for macroinvertebrate image prediction.
    def __init__(self, preprocessor, model_path: Path) -> None:
        super().__init__()
        self.title("Macroinvertebrate Image Analysis System")
        self.geometry("900x600")
        self.preprocessor = preprocessor
        # self.model = joblib.load(model_path)
        self.model = None
        self.selected_file = None
        self.image_label = tk.Label(self, text="No image selected")
        self.image_label.pack(pady=10)
        self.result_label = tk.Label(self, text="Prediction will appear here", font=("Arial", 14))
        self.result_label.pack(pady=10)
        tk.Button(self, text="Choose Image",
        command=self.choose_image).pack(pady=5)
        tk.Button(self, text="Predict",
        command=self.predict_image).pack(pady=5)

    def choose_image(self) -> None:
        # Open a file dialog and preview the selected image.
        file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")] # USE CONFIG?
        )
        if not file_path:
            return
        self.selected_file = file_path
        image = Image.open(file_path)
        image.thumbnail((350, 350))
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

    def predict_image(self) -> None:
        # Predict the class of the currently selected image.
        if not self.selected_file:
            messagebox.showwarning("No image", "Please choose an image first.")
            return
        features = self.preprocessor.transform(self.selected_file).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(features).max()
            text = f"Predicted class: {prediction} | Confidence: {probability:.2%}"
        else:
            text = f"Predicted class: {prediction}"
            self.result_label.configure(text=text)
