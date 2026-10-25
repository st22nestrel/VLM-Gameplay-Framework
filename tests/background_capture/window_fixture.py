"""Small, controllable Win32 test windows; run in a separate process."""

import ctypes
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk

import win32con
import win32gui

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.window_session import focus_window


def main():
    ctypes.windll.user32.SetProcessDPIAware()
    root = tk.Tk()
    root.title("Background capture test target")
    root.geometry("320x240+100+150")
    canvas = tk.Canvas(root, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    def paint(color):
        canvas.delete("all")
        canvas.create_rectangle(0, 0, 160, 240, fill=color, outline=color)
        canvas.create_rectangle(160, 0, 320, 240, fill="#2040c0", outline="#2040c0")

    paint("#20c040")
    cover = tk.Toplevel(root)
    cover.title("Background capture test cover")
    cover.overrideredirect(True)
    cover.configure(bg="#e020e0")
    cover.attributes("-topmost", True)
    cover.withdraw()
    root.update()
    hwnd = win32gui.GetAncestor(root.winfo_id(), win32con.GA_ROOT)
    pygame = None
    if "--opengl" in sys.argv:
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        import pygame

        root.withdraw()
        os.environ["SDL_VIDEO_WINDOW_POS"] = "100,150"
        pygame.display.init()
        pygame.display.set_mode((320, 240), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Background capture OpenGL test target")
        hwnd = pygame.display.get_wm_info()["window"]
        gl = ctypes.windll.opengl32
        gl.glClearColor.argtypes = [ctypes.c_float] * 4

        def paint(color):
            rgb = tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
            gl.glDisable(0x0C11)  # GL_SCISSOR_TEST
            gl.glClearColor(32 / 255, 64 / 255, 192 / 255, 1)
            gl.glClear(0x4000)  # GL_COLOR_BUFFER_BIT
            gl.glEnable(0x0C11)
            gl.glScissor(0, 0, 160, 240)
            gl.glClearColor(*rgb, 1)
            gl.glClear(0x4000)
            pygame.display.flip()

        paint("#20c040")
    cover_hwnd = win32gui.GetAncestor(cover.winfo_id(), win32con.GA_ROOT)
    commands = queue.Queue()

    def read_commands():
        for line in sys.stdin:
            commands.put(line.strip())
        commands.put("quit")

    threading.Thread(target=read_commands, daemon=True).start()
    print(json.dumps({"hwnd": hwnd, "cover_hwnd": cover_hwnd}), flush=True)
    while True:
        root.update()
        if pygame:
            pygame.event.pump()
        try:
            command = commands.get_nowait()
        except queue.Empty:
            time.sleep(0.005)
            continue
        if command == "quit":
            break
        if command in ("partial", "full"):
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left if command == "full" else (right - left) // 2
            cover.geometry(f"{width}x{bottom - top}+{left}+{top}")
            cover.deiconify()
            root.update()
            # Only activate our own test cover, never the target during capture.
            if not focus_window(cover_hwnd):
                raise RuntimeError("Could not activate the test cover on this desktop")
        elif command == "change":
            paint("#e0b020")
        elif command == "minimize":
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        root.update()
        print(json.dumps({"done": command}), flush=True)
    root.destroy()
    if pygame:
        pygame.quit()


if __name__ == "__main__":
    main()
