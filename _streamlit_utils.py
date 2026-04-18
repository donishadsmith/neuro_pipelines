import logging, tkinter as tk
from tkinter import filedialog

class StreamlitLogHandler(logging.Handler):
    def __init__(self, status):
        super().__init__()
        self.status = status
        self.setLevel(logging.INFO)

    def emit(self, record):
        if record.levelno == logging.INFO:
            self.status.write(record.getMessage())

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
