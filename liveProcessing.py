# Grab images from URL and process them for our network

from PIL import Image
import requests
from io import BytesIO
import time
#import twitch
import livestreamer
import cv2

##################################

#streamName  = "esl_sc2"
streamName  = "naniwasc2"
imageWidth  = 1920
imageHeight = 1080
URL         = "https://static-cdn.jtvnw.net/previews-ttv/live_user_" + streamName + "-" + str(imageWidth) + "x" + str(imageHeight) + ".jpg"
streamImageBox = (27, 807, 287, 1066)

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

i = 0
while i < 1:

	# Twitch app stuff
	#client = twitch.TwitchClient(client_id=CLIENT_ID)

	# Livestreamer download
	print("Connecting to the Stream...")
	session = livestreamer.Livestreamer()
	session.set_option("http-headers","Client-ID=" + CLIENT_ID_TWITCH_WEBPLAYER)
	#URL = "https://api.twitch.tv/api/channels/" + streamName + "?client_id=" + CLIENT_ID_TWITCH_WEBPLAYER
	#streams = session.streams(URL)
	#stream = streams['best']

	plugin = session.resolve_url("http://twitch.tv/" + streamName)
	streams = plugin.get_streams()
	stream = streams['best']
	print("Connected to Stream.")

	print("Gathering Stream Data...")
	fd = stream.open()
	data = fd.read(10024)
	fd.close()

	fname = 'stream.bin'
	f = open(fname, 'wb')
	f.write(data)
	f.close()
	time.sleep(5)
	print("Stream Data Gathered.")

	print("Capturing Image...")
	print("open cv before")
	capture = cv2.VideoCapture(fname)
	print("open cv after")
	j = 0
	while capture.isOpened():
		ret, frame = capture.read()
		if ret == True:
			imgdata = frame[...,::-1]
			img = Image.fromarray(imgdata)
			img.save('StreamImages/frame' + str(j) + '.png')
			print("Image Captured and Saved.")
		else:
			break
	#imgdata = capture.read()[1]
	capture.release()

	# response = requests.get(URL)
	# streamImage = Image.open(BytesIO(response.content))

	# croppedImage = streamImage.crop(streamImageBox)

	# paddedImage = createPaddedImage(croppedImage, imageResize)

	# #TODO: figure out a way to see if you won by taking pictures of the location of the victory message

	# paddedImage.save(destinationFolder+saveImageName+str(i)+".png")
	i = i + 1