import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread, Event

# Add the src directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.capture.video_capture import capture_video

class BlurToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Blur Tool for IRL Streaming")

        self.video_source = tk.StringVar()
        self.output_file = tk.StringVar()
        self.stop_event = Event()
        self.capture_thread = None

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Video Source:").grid(row=0, column=0, padx=10, pady=10)
        tk.Entry(self.root, textvariable=self.video_source).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.browse_video_source).grid(row=0, column=2, padx=10, pady=10)

        tk.Label(self.root, text="Output File:").grid(row=1, column=0, padx=10, pady=10)
        tk.Entry(self.root, textvariable=self.output_file).grid(row=1, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.browse_output_file).grid(row=1, column=2, padx=10, pady=10)

        tk.Button(self.root, text="Start", command=self.start_blur_tool).grid(row=2, column=0, columnspan=3, pady=20)
        tk.Button(self.root, text="Stop", command=self.stop_blur_tool).grid(row=3, column=0, columnspan=3, pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def browse_video_source(self):
        file_path = filedialog.askopenfilename()
        self.video_source.set(file_path)

    def browse_output_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
        self.output_file.set(file_path)

    def start_blur_tool(self):
        video_source = self.video_source.get()
        output_file = self.output_file.get()
        if video_source == "":
            video_source = "0"  # Default to webcam if no source is provided

        print(f"Starting blur tool with video source: {video_source}, output file: {output_file}")
        if video_source and output_file:
            self.stop_event.clear()
            self.capture_thread = Thread(target=capture_video, args=(video_source, output_file, self.stop_event))
            self.capture_thread.start()
        else:
            messagebox.showerror("Error", "Please specify both video source and output file.")

    def stop_blur_tool(self):
        if self.capture_thread and self.capture_thread.is_alive():
            self.stop_event.set()
            self.capture_thread.join()
            print("Blur tool stopped.")

    def on_closing(self):
        self.stop_blur_tool()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = BlurToolApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
