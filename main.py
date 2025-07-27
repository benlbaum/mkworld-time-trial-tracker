from tkinter import Tk
from time_trial_tracker.gui import LapTimerGUI

def main():
    root = Tk()
    app = LapTimerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
