from PIL import ImageGrab
import win32gui
import time

toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(enum_cb, toplist)

starcraft2Client = [(hwnd, title) for hwnd, title in winlist if 'starcraft ii' in title.lower()]
# just grab the hwnd for first window matching starcraft2Client
if len(starcraft2Client) == 0:
    print("[ERROR] Starcraft 2 Client Not Running.")
    exit()
starcraft2Client = starcraft2Client[0]
hwnd = starcraft2Client[0]

win32gui.SetForegroundWindow(hwnd)
#time.sleep(.1)
bbox = win32gui.GetWindowRect(hwnd)
print(bbox)
bbox = (6,913,250,1160)
img = ImageGrab.grab(bbox)
#img.show()
img.save("images/test.png")