import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk
import threading

from Services.eda_service import EDAService

class MacroApp(tk.Tk):
    # INITIALISE
    def __init__(self, image_folder: Path, start_callback=None) -> None:
        super().__init__()

        self.title("Macroinvertebrate Visualisation System")
        self.geometry("1200x750")
        self.minsize(1000, 650)
        self.start_callback = start_callback

        self.protocol("WM_DELETE_WINDOW", self.on_close)


        ## Fullscreen windowed
        self.update_idletasks()

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        self.after(0, lambda: self.geometry(
            f"{screen_width}x{screen_height}+0+0"
        ))

        ## DATA
        self.image_folder = image_folder
        self.image_files = []
        self.reload_images()

        self.image_folder = image_folder
        self.current_index = 0
        self.current_photo = None


        # MAIN LAYOUT
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        ## LEFT SIDEBAR
        self.sidebar = tk.Frame(self, bg="#2B2B2B", width=260)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)

        title = tk.Label(
            self.sidebar,
            text="Controls",
            bg="#2B2B2B",
            fg="white",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=20)

        # START BUTTON
        self.start_button = tk.Button(
            self.sidebar,
            text="Start Indexing",
            command=self.start_processing,
            width=22,
            height=3
        )
        self.start_button.pack(pady=10)

        # PROGRESS BAR
        self.progress = ttk.Progressbar(
            self.sidebar,
            orient="horizontal",
            length=200,
            mode="determinate"
        )

        self.progress.pack(pady=10)

        # NEXT BUTTON
        self.next_button = tk.Button(
            self.sidebar,
            text="Next Image",
            command=self.next_image,
            width=22,
            height=3
        )
        self.next_button.pack(pady=10)

        # PREVIOUS BUTTON
        self.prev_button = tk.Button(
            self.sidebar,
            text="Previous Image",
            command=self.previous_image,
            width=22,
            height=3
        )
        self.prev_button.pack(pady=10)

        ## RIGHT CONTENT AREA
        self.content_frame = tk.Frame(self, bg="#EAEAEA")
        self.content_frame.grid(row=0, column=1, sticky="nsew")


        ## IMAGE VIEWER
        self.viewer_frame = tk.Frame(
            self.content_frame,
            bg="black",
            bd=2,
            relief="sunken"
        )

        self.viewer_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20,
            ipadx=0,
            ipady=0
        )

        self.viewer_frame.configure(height=450)

        self.image_label = tk.Label(
            self.viewer_frame,
            text="Press 'Start Indexing'",
            bg="black",
            fg="white",
            font=("Arial", 14)
        )

        self.image_label.pack(fill="both", expand=True)

        ## STATUS LABEL
        self.status_label = tk.Label(
            self.content_frame,
            text="Waiting to start...",
            font=("Arial", 13),
            bg="#EAEAEA"
        )

        self.status_label.pack(pady=(0, 10))

        ## TEXT DISPLAY AREA
        self.text_frame = tk.Frame(
            self.content_frame,
            bg="#D8D8D8"
        )

        self.text_frame.pack(
            fill="both",
            padx=20,
            pady=(0, 20)
        )

        scrollbar = tk.Scrollbar(self.text_frame)
        scrollbar.pack(side="right", fill="y")

        self.text_box = tk.Text(
            self.text_frame,
            height=8,
            wrap="word",
            font=("Arial", 11),
            yscrollcommand=scrollbar.set
        )

        self.text_box.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.config(command=self.text_box.yview)

        self.text_box.insert(
            "1.0",
            "Related text will appear here."
        )

        self.text_box.config(state="disabled")

    ## START PROCESSING
    def start_processing(self) -> None:
        self.start_button.config(state="disabled")
        self.prev_button.config(state="disabled")
        self.next_button.config(state="disabled")

        if self.start_callback is None:
            return

        self.progress["value"] = 0
        self.status_label.configure(text="Starting dataset indexing...")

        thread = threading.Thread(
            target=self.run_indexing,
            daemon=True
        )

        thread.start()

    # RUN INDEXER
    def run_indexing(self):
        try:
            df, output_dir = self.start_callback(
                progress_callback=lambda c, t: self.after(
                    0,
                    lambda: self.update_progress(c, t)
                )
            )

            # now safely pass to main thread
            self.after(0, lambda: self.on_indexing_complete(df, output_dir))

        except Exception as e:
            self.after(0, lambda: self.status_label.configure(
                text=f"Error: {e}"
            ))

    # PROGRESS BAR
    def update_progress(self, current, total):
        percentage: float
        percentage = (current / total) * 100

        self.progress["value"] = percentage

        self.status_label.configure(
            text=f"Processing {current}/{total} images..."
        )

        self.update_idletasks()

    # REFRESH DISPLAYED IMAGE
    def reload_images(self):
        if self.image_folder.exists():

            self.image_files = sorted([
                file for file in self.image_folder.iterdir()
                if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ])

        else:
            self.image_files = []

    # RUN EDA FUNCTIONS
    def generate_eda(self, df, output_dir):
        try:
            self.status_label.configure(text="Generating EDA visualisations...")

            eda_service = EDAService(
                dataframe=df,
                output_dir=output_dir
            )

            eda_service.generate_all_outputs()

            summary = eda_service.build_summary()

            print(summary)

            self.status_label.configure(text="EDA complete.")

        except Exception as e:
            self.status_label.configure(text=f"EDA error: {e}")

    # CLEANUP
    def on_indexing_complete(self, df, output_dir):
        # Reset progress bar
        self.progress["value"] = 0

        # Update status
        self.status_label.configure(
            text="Indexing complete."
        )

        # Refresh image list (IMPORTANT if dataset changed)
        self.reload_images()

        # Load first image automatically
        self.current_index = 0
        self.after(0, self.load_current_image)


        self.start_button.config(state="normal")
        self.prev_button.config(state="normal")
        self.next_button.config(state="normal")

        # RUN EDA
        self.generate_eda(df, output_dir)


    ## IMAGE DISPLAY
    # LOAD CURRENT IMAGE
    def load_current_image(self) -> None:
        image: Image.image

        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]

        with Image.open(image_path) as image:
            self.display_image(image.copy())

        self.display_image(image)

        self.status_label.configure(
            text=f"Viewing: {image_path.name}"
        )

        self.load_related_text(image_path)

    # DISPLAY IMAGE
    def display_image(self, image: Image.Image) -> None:
        # Update frame dimensions
        self.viewer_frame.update_idletasks()

        frame_width = int(self.viewer_frame.winfo_width() * 0.9)
        frame_height = int(self.viewer_frame.winfo_height() * 0.9)

        # Prevent invalid startup sizes
        if frame_width < 10:
            frame_width = 800

        if frame_height < 10:
            frame_height = 500

        # Resize image to fill frame
        resized_image = image.copy()
        resized_image.thumbnail(
            (frame_width, frame_height),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(resized_image)

        self.image_label.configure(
            image=photo,
            text=""
        )

        self.image_label.image = photo
        self.current_photo = photo

    # LOAD RELATED TEXT
    def load_related_text(self, image_path: Path) -> None:
        # Search for text file with name matching image
        txt_path = image_path.with_suffix(".txt")

        self.text_box.config(state="normal")
        self.text_box.delete("1.0", tk.END)

        if txt_path.exists():

            try:
                text_content = txt_path.read_text(
                    encoding="utf-8"
                )

                self.text_box.insert(
                    "1.0",
                    text_content
                )

            except Exception as e:

                self.text_box.insert(
                    "1.0",
                    f"Error reading text file:\n{e}"
                )

        else:

            self.text_box.insert(
                "1.0",
                "No related text file found."
            )

        self.text_box.config(state="disabled")
        
    def on_close(self):

        self.quit()
        self.destroy()


    # SHOW NEXT IMAGE
    def next_image(self) -> None:
        if not self.image_files:
            return

        self.current_index = (
            self.current_index + 1
        ) % len(self.image_files)

        self.load_current_image()

    # SHOW PREVIOUS IMAGE
    def previous_image(self) -> None:
        if not self.image_files:
            return

        self.current_index = (
            self.current_index - 1
        ) % len(self.image_files)

        self.load_current_image()

