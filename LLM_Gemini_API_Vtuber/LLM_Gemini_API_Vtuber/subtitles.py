import tkinter as tk

class SubtitleOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sottotitoli VTuber")
        chroma_key = "#010101" 
        self.root.config(bg=chroma_key)
        self.root.wm_attributes("-transparentcolor", chroma_key)
        self.root.attributes("-topmost", True)                                              # Sempre in primo piano
        self.root.overrideredirect(True)        

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width+350}x150+0+{screen_height - 30}")

        self.label = tk.Label(
            self.root, 
            text="", 
            font=("Helvetica", 16, "bold"), 
            fg="white",         
            bg=chroma_key,        
            wraplength=screen_width - 100, 
            justify="center"
        )
        self.label.pack(expand=True)

    def toggle_subtitles(self, state: bool):
        self.active = state
        if self.active:
            self.root.deiconify() 
            self.root.attributes("-topmost", True) 
        else:
            self.root.withdraw() 

    def update_text(self, new_text):
        if not self.active:
            return
        self.label.config(text=new_text)
        self.root.update_idletasks() 
        self.root.update()

    def clear_text(self):
        self.label.config(text="")
        if self.active:
            self.root.update()