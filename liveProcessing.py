# Grab images from URL and process them for our network

from PIL import Image
import requests
from io import BytesIO
import time
#import twitch #pip install python-twitch-client
import livestreamer
import cv2

##################################

tournementStreamNames = ["esl_sc2","starcraft"]

streamName  = "esl_sc2"
#streamName  = "kennystream"
imageWidth  = 1920
imageHeight = 1080
#URL         = "https://static-cdn.jtvnw.net/previews-ttv/live_user_" + streamName + "-" + str(imageWidth) + "x" + str(imageHeight) + ".jpg"
if streamName in tournementStreamNames:
	streamImageBox = (0,  842, 245, 1079) #ESL stream
else:
	#its a personal stream
	streamImageBox = (27, 807, 287, 1066) #regular streamer

imageResize = 256
destinationFolder = "StreamImages/" + streamName + "/"

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

def inGame(streamIm, streamName, tournementStreamNames):
	in_game = False

	if streamName in tournementStreamNames:
		#check the blue and red colors on the interface for names

		#Blue Player pixel
		bluePixelGood = False
		redPixelGood = False
		r,g,b = streamIm.getpixel((531,991))
		if (r < 30 and b > 200):
			bluePixelGood = True
		if (r > 205 and b < 30):
			redPixelGood = True

		#Red Player pixel
		r,g,b = streamIm.getpixel((531,1058))
		if bluePixelGood:
			if (r > 205 and b < 30):
				redPixelGood = True
		if redPixelGood:
			if (r < 30 and b > 200):
				bluePixelGood = True

		if bluePixelGood and redPixelGood:
			in_game = True
	else:
		#check pixels for mineral and gas
		pass

	return in_game

##################################

#print("Preview Stream URL")
#print(URL)
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
while True:

	# Twitch app stuff
	#client = twitch.TwitchClient(client_id=CLIENT_ID)

	print("Gathering Stream Data...")
	fd = stream.open()
	data = fd.read(400024)
	fd.close()

	fname = 'stream.mp4'
	f = open(fname, 'wb')
	f.write(data)
	f.close()
	print("Stream Data Gathered.")

	print("Capturing Image...")
	capture = cv2.VideoCapture(fname)

	frameImage = 'frame' + str(i) + '.png'

	ret, frame = capture.read()
	if ret == True:
		#Image was captured, lets use it
		imgdata = frame[...,::-1]
		img = Image.fromarray(imgdata)
		if inGame(img, streamName, tournementStreamNames):
			#we are actually in a game, lets take the map image
			print("IN GAME")
			#img.save(destinationFolder+frameImage) #save the full screen image
			croppedImage   = img.crop(streamImageBox)
			paddedImage    = createPaddedImage(croppedImage, imageResize)
			paddedImage.save(destinationFolder+"cropped"+frameImage)
			i = i + 1	
			print("Image Captured and Saved.")
		else:
			print("NOT GAME")
	else:
		print("Imaged Failed to Captured.")
	capture.release()
	# #TODO: figure out a way to see if you won by taking pictures of the location of the victory message

	