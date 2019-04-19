# Grab images from URL and process them for our network

from PIL import Image
import requests
from io import BytesIO
import time
#import twitch #pip install python-twitch-client
import livestreamer
import cv2

##################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import os
import sys
from torchvision import transforms
import datetime as dt
from absl import flags
#from fastai.vision.image import open_image

##################################

FLAGS = flags.FLAGS
flags.DEFINE_string("stream", None, "The Twitch stream to observe. I.E. \"esl_sc2\"")
flags.DEFINE_string("model", None, "The PyTorch model to use.")
flags.DEFINE_bool("save_images", False, "Whether to save the stream map images or not.")

flags.mark_flag_as_required("stream")
flags.mark_flag_as_required("model")

FLAGS(sys.argv)

##################################

#modelPath = "Models/ResNet18_BinaryClassification_FullVision/"
#modeFileName = "model_00_0.pth"
modelPath = FLAGS.model

tournementStreamNames = ["esl_sc2","starcraft"]

#streamName  = "esl_sc2"
#streamName  = "kennystream"
streamName = FLAGS.stream

imageWidth  = 1920
imageHeight = 1080

if streamName in tournementStreamNames:
	streamImageBox = (0,  842, 245, 1079) #ESL stream
else:
	#its a personal stream
	streamImageBox = (27, 807, 287, 1066) #regular streamer

imageResize = 256

destinationFolder = "StreamImages/" + streamName + "/"
if not(os.path.isdir("StreamImages/")):
	os.mkdir("StreamImages")
if not(os.path.isdir("StreamImages/" + streamName + "/")):
	os.mkdir("StreamImages/" + streamName)

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

def live_plotter(x_vec,y1_data,lines,pause_time=0.1):
    if lines==[]:
    	# this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(-3,3)
        # create a variable for the line so we can later update it
        lines, = ax.plot(x_vec,y1_data,alpha=0.8)        
        #update plot label/title
        plt.ylabel('Prediction')
        plt.xlabel('Time')
        plt.title('Outcome Prediction Over time')
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    lines.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=lines.axes.get_ylim()[0] or np.max(y1_data)>=lines.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return lines

##################################

#print("Preview Stream URL")
#print(URL)

# Load model class
#model = torch.load(modelPath+modeFileName)
model = torch.load(modelPath)
model.eval()

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

####################################
normalize = transforms.Normalize(
	mean=[0.1213, 0.1105, 0.1275],
	std=[0.1201, 0.1079, 0.1408]
)

preprocess = transforms.Compose([
   transforms.ToTensor(),
   normalize
])

# Create figure for plotting
plt.style.use('ggplot')
size = 50
xs = np.linspace(0,1,size+1)[0:-1]
ys = np.linspace(0,0,size+1)[0:-1]
lines = []

i = 0
while True:
	#print("Gathering Stream Data...")
	fd = stream.open()
	data = fd.read(400024)
	fd.close()

	fname = 'stream.mp4'
	f = open(fname, 'wb')
	f.write(data)
	f.close()
	#print("Stream Data Gathered.")

	#print("Capturing Image...")
	capture = cv2.VideoCapture(fname)

	frameImage = 'frame' + str(i) + '.png'

	ret, frame = capture.read()
	if ret == True:
		#Image was captured, lets use it
		imgdata = frame[...,::-1]
		img = Image.fromarray(imgdata)
		#if inGame(img, streamName, tournementStreamNames):
		#we are actually in a game, lets take the map image
		print("IN GAME")
		#img.save(destinationFolder+frameImage) #save the full screen image
		croppedImage   = img.crop(streamImageBox)
		paddedImage    = createPaddedImage(croppedImage, imageResize)

		if FLAGS.save_images == True:
			paddedImage.save(destinationFolder+"resized"+frameImage)
		imgTensor = preprocess(paddedImage)
		imgTensor.unsqueeze_(0)

		output = model(imgTensor.cuda())
		print("TENSOR: " + str(output))
		print("OUTPUT: " + str(output[0][0].item()))

		# Add x and y to lists
		#xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
		team1WinPred = output[0][0].item()
		team2WinPred = output[0][1].item()
		value = 0
		if team2WinPred > team1WinPred:
			# TODO: decide if this subtraction is needed. The 
			# Thought behind it is if both are equally likely 
			# to be true, it should be a draw.
			value = -1 * (team2WinPred - team1WinPred)
		else:
			value = (team1WinPred - team2WinPred)
		#xs = np.append(xs,i)
		ys = np.append(ys,value)

		# Limit x and y lists to the last 50 elements
		#xs = xs[-50:]
		ys = ys[-50:]

		lines = live_plotter(xs,ys,lines)

		# if value < 0:
		# 	newsegm, = ax.plot(xs, ys, color='red')
		# else:
		# 	newsegm, = ax.plot(xs, ys, color='blue')

		#else:
		#print("NOT GAME")
	else:
		print("Imaged Failed to Captured.") 

	i += 1


####################################

#ani = animation.FuncAnimation(fig, animate, frames=50, fargs=(stream, xs, ys), interval=1000)
ani = animation.FuncAnimation(fig, animate, frames=50, interval=1000)
plt.show()


#capture.release()
	# #TODO: figure out a way to see if you won by taking pictures of the location of the victory message