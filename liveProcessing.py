# Grab images from URL and process them for our network

from PIL import Image
import requests
from io import BytesIO
import time
#import twitch #pip install python-twitch-client
import livestreamer
import cv2

##################################

#streamName  = "esl_sc2"
streamName  = "kennystream"
imageWidth  = 1920
imageHeight = 1080
URL         = "https://static-cdn.jtvnw.net/previews-ttv/live_user_" + streamName + "-" + str(imageWidth) + "x" + str(imageHeight) + ".jpg"
streamImageBox = (27, 807, 287, 1066) #regular streamer
#streamImageBox = (0,  842, 245, 1079) #ESL stream

imageResize = 256
destinationFolder = "StreamImages/" + streamName + "/"

CLIENT_ID = "fh146rt2ojrmmgopnzuvplfe7v4yyh" #PERSONAL
CLIENT_ID_TWITCH_WEBPLAYER = "jzkbprff40iqj646a697cyrvl0zt2m6"

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

print("Preview Stream URL")
print(URL)
print("------------------")

# Livestreamer download
print("Connecting to the Stream...")
session = livestreamer.Livestreamer()
session.set_option("http-headers","Client-ID=" + CLIENT_ID_TWITCH_WEBPLAYER)
#URL = "https://api.twitch.tv/api/channels/" + streamName + "?client_id=" + CLIENT_ID_TWITCH_WEBPLAYER
#streams = session.streams(URL)
#stream = streams['best']

plugin = session.resolve_url("http://twitch.tv/" + streamName)
streams = plugin.get_streams()
#stream = streams['best']
stream = streams['1080p60']
print("Connected to Stream.")

i = 0
while i < 5:

	# Twitch app stuff
	#client = twitch.TwitchClient(client_id=CLIENT_ID)

	print("Gathering Stream Data...")
	fd = stream.open()
	data = fd.read(300024)
	fd.close()

	fname = 'stream.mp4'
	f = open(fname, 'wb')
	f.write(data)
	f.close()
	print("Stream Data Gathered.")

	print("Capturing Image...")
	capture = cv2.VideoCapture(fname)

	frameImage = 'frame' + str(i) + '.png'
	imageCaptured = False
	while capture.isOpened():
		ret, frame = capture.read()
		if ret == True:
			imgdata = frame[...,::-1]
			img = Image.fromarray(imgdata)
			img.save(destinationFolder+frameImage)
			print("Image Captured and Saved.")
			imageCaptured = True
			break
		else:
			print("Imaged Failed to Captured.")
			imageCaptured = False
			break
	#imgdata = capture.read()[1]
	capture.release()

	if imageCaptured:
		streamImageObj = Image.open(destinationFolder+frameImage)
		croppedImage   = streamImageObj.crop(streamImageBox)
		streamImageObj.close()
		paddedImage    = createPaddedImage(croppedImage, imageResize)
		paddedImage.save(destinationFolder+"cropped"+frameImage)

		# streamImage = Image.open(BytesIO(response.content))
		# croppedImage = streamImage.crop(streamImageBox)
		# paddedImage = createPaddedImage(croppedImage, imageResize)
		# paddedImage.save(destinationFolder+saveImageName+str(i)+".png")

	# #TODO: figure out a way to see if you won by taking pictures of the location of the victory message

	i = i + 1