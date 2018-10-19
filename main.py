import win32gui
import win32ui
from ctypes import windll
from PIL import Image

def WindowExists(windowname):
    try:
        win32ui.FindWindow(None, windowname)

    except win32ui.error:
        return False
    else:
        return True

#PROGRAM = "MATLAB R2018b - prerelease use"
PROGRAM = "Starcraft II"

if not(WindowExists(PROGRAM)):
	print("The program '" + PROGRAM + "' does not exist.")
	exit()


hwnd = win32gui.FindWindow(None, PROGRAM)

# Change the line below depending on whether you want the whole window
# or just the client area. 
#left, top, right, bot = win32gui.GetClientRect(hwnd)
left, top, right, bot = win32gui.GetWindowRect(hwnd)
w = right - left
h = bot - top

hwndDC = win32gui.GetWindowDC(hwnd)
mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
saveDC = mfcDC.CreateCompatibleDC()

saveBitMap = win32ui.CreateBitmap()
saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

saveDC.SelectObject(saveBitMap)

# Change the line below depending on whether you want the whole window
# or just the client area. 
#result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
print(result)

if result == 1:
	#PrintWindow Succeeded
	bmpinfo = saveBitMap.GetInfo()
	bmpstr = saveBitMap.GetBitmapBits(True)

	im = Image.frombuffer(
	    'RGB',
	    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
	    bmpstr, 'raw', 'BGRX', 0, 1)   

	im.save("test.png")

win32gui.DeleteObject(saveBitMap.GetHandle())
saveDC.DeleteDC()
mfcDC.DeleteDC()
win32gui.ReleaseDC(hwnd, hwndDC) 