import os
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import json

# Function to load theme settings from settings.json
def load_theme_settings():
    try:
        with open('settings.json', 'r') as file:
            settings = json.load(file)
            return settings.get('lastTheme', 'Default')
    except FileNotFoundError:
        messagebox.showwarning("Settings File Not Found", "Could not find 'settings.json'. Using default theme.")
        return 'Default'

# Function to apply the stylesheet to Tkinter elements (basic conversion from QSS to Tkinter compatible style)
def apply_stylesheet_to_tkinter(master, theme_name):
    if theme_name == "Default":
        return  # Do nothing, default style

    # Assuming you want to load and parse a .qss or .css file manually and extract color properties
    style_folder = os.path.join(os.getcwd(), 'styles')
    theme_file_path = os.path.join(style_folder, theme_name)

    try:
        with open(theme_file_path, 'r', encoding="utf-8") as f:
            stylesheet = f.read()

        # Parsing .qss or .css for basic properties (you can expand this as needed)
        background_color = None
        foreground_color = None

        # Basic parsing for background and foreground color
        for line in stylesheet.splitlines():
            if 'background-color' in line:
                background_color = line.split(':')[-1].strip().strip(';')
            if 'color' in line and 'background-color' not in line:
                foreground_color = line.split(':')[-1].strip().strip(';')

        # Apply the parsed colors to the Tkinter window and widgets
        if background_color:
            master.configure(bg=background_color)
        return {"background_color": background_color, "foreground_color": foreground_color}

    except Exception as e:
        print(f"Failed to load or apply stylesheet: {e}")
        return {}

class DatasetSplitterApp:
    def __init__(self, master, theme_colors):
        self.master = master
        self.master.title("Dataset Splitter")

        # Apply background color from the theme (if available)
        self.master.configure(bg=theme_colors.get("background_color", "#ffffff"))

        # Tkinter UI elements
        self.dataset_path = None

        # Dataset path
        self.dataset_label = tk.Label(master, text="Dataset Directory:", bg=theme_colors.get("background_color"), fg=theme_colors.get("foreground_color"))
        self.dataset_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)

        self.dataset_entry = tk.Entry(master, width=50, bg=theme_colors.get("background_color"), fg=theme_colors.get("foreground_color"))
        self.dataset_entry.grid(row=0, column=1, padx=10, pady=5)

        self.browse_button = tk.Button(master, text="Browse", command=self.select_dataset)
        self.browse_button.grid(row=0, column=2, padx=10, pady=5)

        # Chunk size
        self.chunk_label = tk.Label(master, text="Chunk Size:", bg=theme_colors.get("background_color"), fg=theme_colors.get("foreground_color"))
        self.chunk_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)

        self.chunk_entry = tk.Entry(master, width=10)
        self.chunk_entry.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        self.chunk_entry.insert(0, "10000")  # Default chunk size

        # Split button
        self.split_button = tk.Button(master, text="Split Dataset", command=self.split_dataset)
        self.split_button.grid(row=2, column=1, padx=10, pady=20)

        # Revert button
        self.revert_button = tk.Button(master, text="Revert Split", command=self.revert_split)
        self.revert_button.grid(row=2, column=2, padx=10, pady=20)

    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_path:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, self.dataset_path)

    def split_dataset(self):
        dataset_dir = self.dataset_entry.get()
        chunk_size = int(self.chunk_entry.get())

        if not dataset_dir:
            messagebox.showwarning("Missing Information", "Please select a dataset directory.")
            return

        dataset_path = Path(dataset_dir)
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
        image_files = [f for f in dataset_path.iterdir() if f.suffix.lower() in image_extensions]
        image_files.sort()

        # Path to the original classes.txt
        classes_file = dataset_path / "classes.txt"
        if not classes_file.exists():
            messagebox.showwarning("File Not Found", "classes.txt not found in the dataset directory.")
            return

        def move_files_to_folder(files, folder_name):
            folder_path = dataset_path / folder_name
            folder_path.mkdir(exist_ok=True)

            # Copy classes.txt into the new folder
            shutil.copy(str(classes_file), str(folder_path / "classes.txt"))

            for image_file in files:
                shutil.move(str(image_file), str(folder_path / image_file.name))

                # Check for corresponding text files and move them
                txt_file = dataset_path / (image_file.stem + ".txt")
                if txt_file.exists():
                    shutil.move(str(txt_file), str(folder_path / txt_file.name))

        # Split dataset into chunks
        for i in range(0, len(image_files), chunk_size):
            chunk = image_files[i:i + chunk_size]
            folder_name = f"images_{i // chunk_size + 1}"
            move_files_to_folder(chunk, folder_name)

        messagebox.showinfo("Success", "Dataset successfully split into folders!")

    def revert_split(self):
        dataset_dir = self.dataset_entry.get()

        if not dataset_dir:
            messagebox.showwarning("Missing Information", "Please select a dataset directory.")
            return

        dataset_path = Path(dataset_dir)
        folder_prefix = "images_"

        # Find folders starting with "images_"
        split_folders = [f for f in dataset_path.iterdir() if f.is_dir() and f.name.startswith(folder_prefix)]

        for folder in split_folders:
            # Move the files and subfolders back to the original dataset directory
            for item in folder.iterdir():
                if item.is_file():
                    if item.name == "classes.txt":
                        # Only move classes.txt if it doesn't exist in the dataset directory
                        dest_classes_file = dataset_path / "classes.txt"
                        if not dest_classes_file.exists():
                            shutil.move(str(item), str(dest_classes_file))
                        else:
                            # If classes.txt already exists, remove the duplicate
                            item.unlink()
                    else:
                        # Move other files back to the dataset directory
                        shutil.move(str(item), str(dataset_path / item.name))
                elif item.is_dir():
                    # If the item is a directory (e.g., "thumbnails"), we need to move its contents
                    dest_subfolder = dataset_path / item.name
                    dest_subfolder.mkdir(exist_ok=True)
                    for subitem in item.iterdir():
                        shutil.move(str(subitem), str(dest_subfolder / subitem.name))
                    # After moving contents, remove the subfolder if empty
                    if not any(item.iterdir()):
                        item.rmdir()
            # Remove the images_n folder if it's empty
            if not any(folder.iterdir()):
                folder.rmdir()

        messagebox.showinfo("Success", "Dataset successfully reverted and split folders removed!")

# Main function to start Tkinter app
def start_tkinter_app():
    root = tk.Tk()

    # Load the last used theme from settings.json
    last_theme = load_theme_settings()

    # Apply the stylesheet to Tkinter
    theme_colors = apply_stylesheet_to_tkinter(root, last_theme)

    # Create the DatasetSplitterApp with theme colors
    DatasetSplitterApp(root, theme_colors)

    root.mainloop()


if __name__ == "__main__":
    start_tkinter_app()

