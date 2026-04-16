import tkinter as tk
from tkinter import filedialog


def _select_content(content):
    window = tk.Tk()
    window.withdraw()
    window.wm_attributes("-topmost", 1)
    if content == "directory":
        folder = filedialog.askdirectory(master=window)
    else:
        folder = filedialog.askopenfilename(master=window)

    window.destroy()

    return folder
