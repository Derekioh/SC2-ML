try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

print(pytesseract.image_to_string(Image.open("timeExample.png")))

print(pytesseract.image_to_string(Image.open("mineralExampleCropped.png").convert("LA")))

print(pytesseract.image_to_string(Image.open("supplyExample.png")))