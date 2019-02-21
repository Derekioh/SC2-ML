# Grab images from URL and process them for our network

from PIL import Image
import requests
from io import BytesIO
import time

##################################

streamName  = "kimdaeyeob3"
imageWidth  = 1920
imageHeight = 1080
URL         = "https://static-cdn.jtvnw.net/previews-ttv/live_user_" + streamName + "-" + str(imageWidth) + "x" + str(imageHeight) + ".jpg"
streamImageBox = (27, 807, 287, 1066)

imageResize = 256
destinationFolder = "StreamImages/kimdaeyeob3/"

#TODO: check files for images to prevent creating duplicates

saveImageName = streamName + "_"

def createPaddedImage(im, desiredImageSize):
	old_size = im.size  # old_size[0] is in (width, height) format

	ratio = float(desiredImageSize)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	# use thumbnail() or resize() method to resize the input image

	# thumbnail is a in-place operation

	# im.thumbnail(new_size, Image.ANTIALIAS)
	im = im.resize(new_size, Image.ANTIALIAS)

	# create a new image and paste the resized on it
	new_im = Image.new("RGB", (desiredImageSize, desiredImageSize))
	new_im.paste(im, ((desiredImageSize-new_size[0])//2,
	                    (desiredImageSize-new_size[1])//2))

	return new_im

##################################

print(URL)

i = 0
while True:

	response = requests.get(URL)
	streamImage = Image.open(BytesIO(response.content))

	croppedImage = streamImage.crop(streamImageBox)

	paddedImage = createPaddedImage(croppedImage, imageResize)

	#TODO: figure out a way to see if you won by taking pictures of the location of the victory message

	paddedImage.save(destinationFolder+saveImageName+str(i)+".png")
	i = i + 1

	print("saved Image.")

	time.sleep(5)