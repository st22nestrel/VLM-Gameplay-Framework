"""Separate-process message receiver, covered without taking foreground focus."""

import json
import queue
import sys
import threading
import time

import win32api
import win32con
import win32gui


def main():
    events = []
    recipients = set()

    def wndproc(hwnd, message, wparam, lparam):
        # TranslateMessage may also derive WM_CHAR from keydown. Record sender
        # transitions separately from those application-generated text events.
        if hwnd in recipients and (message in (0x100, 0x101, 0x104, 0x105) or 0x200 <= message <= 0x209):
            events.append([hwnd, message, wparam, lparam & 0xFFFFFFFF])
            return 0
        if message == win32con.WM_APP:
            print(json.dumps({"events": events}), flush=True)
            events.clear()
            return 0
        return win32gui.DefWindowProc(hwnd, message, wparam, lparam)

    instance = win32api.GetModuleHandle(None)
    wc = win32gui.WNDCLASS()
    wc.hInstance = instance
    wc.lpszClassName = "BackgroundInputTestReceiver"
    wc.lpfnWndProc = wndproc
    atom = win32gui.RegisterClass(wc)
    windows = []
    try:
        root = win32gui.CreateWindowEx(0, atom, "Background input test target",
                                      win32con.WS_OVERLAPPEDWINDOW, 80, 120, 320, 280,
                                      0, 0, instance, None)
        windows.append(root)
        child = win32gui.CreateWindowEx(0, atom, "Input child", win32con.WS_CHILD | win32con.WS_VISIBLE,
                                       20, 30, 180, 150, root, 0, instance, None)
        recipients.update((root, child))
        win32gui.ShowWindow(root, win32con.SW_SHOWNOACTIVATE)
        left, top, right, bottom = win32gui.GetWindowRect(root)
        cover = win32gui.CreateWindowEx(win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOOLWINDOW,
                                       atom, "Background input test cover", win32con.WS_POPUP,
                                       left, top, right - left, bottom - top, 0, 0, instance, None)
        windows.append(cover)
        win32gui.SetWindowPos(cover, win32con.HWND_TOPMOST, left, top, right - left, bottom - top,
                             win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW)
        print(json.dumps({"hwnd": root, "child": child, "cover": cover}), flush=True)
        commands = queue.Queue()

        def read_commands():
            for line in sys.stdin:
                commands.put(line.strip())
            commands.put("quit")

        threading.Thread(target=read_commands, daemon=True).start()
        while True:
            win32gui.PumpWaitingMessages()
            try:
                command = commands.get_nowait()
            except queue.Empty:
                time.sleep(0.005)
                continue
            if command == "quit":
                break
            if command == "events":
                # Barrier after previously posted inputs in this thread's queue.
                win32gui.PostMessage(root, win32con.WM_APP, 0, 0)
    finally:
        for hwnd in reversed(windows):
            if win32gui.IsWindow(hwnd):
                win32gui.DestroyWindow(hwnd)
        win32gui.UnregisterClass(atom, instance)


if __name__ == "__main__":
    main()
