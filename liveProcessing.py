# Grab images from URL and process them for our network

from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance
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
import threading
import multiprocessing
import os
import sys
import datetime as dt
from absl import flags

# import torch
# from torchvision import transforms

from joblib import dump, load
from skimage.measure import compare_ssim as ssim

##################################

FLAGS = flags.FLAGS
flags.DEFINE_string("stream", None, "The Twitch stream to observe. I.E. \"esl_sc2\"")
#flags.DEFINE_string("model", None, "The PyTorch model to use.")
flags.DEFINE_string("model", None, "The sklearn model we want to use.")
flags.DEFINE_bool("save_images", False, "Whether to save the stream map images or not.")

flags.mark_flag_as_required("stream")
flags.mark_flag_as_required("model")

FLAGS(sys.argv)

##################################

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

CONFIG = 'outputbase digitsMOD'

#OLD SPECTATOR INTERFACE
#top left, bottom right
# TIME_TOP_LEFT = (0,770)
# TIME_BOT_RIGHT = (90,795)

# P1_TOP_LEFT_Y = 960
# P1_BOT_RIGHT_Y = 1008
# P2_TOP_LEFT_Y = 1030
# P2_BOT_RIGHT_Y = 1075

# POP_TOP_LEFT_X = 600
# POP_BOT_RIGHT_X = 740

# MINS_TOP_LEFT_X = POP_BOT_RIGHT_X
# MINS_BOT_RIGHT_X = 845

# GAS_TOP_LEFT_X = MINS_BOT_RIGHT_X
# GAS_BOT_RIGHT_X = 950

# WORKERS_TOP_LEFT_X = GAS_BOT_RIGHT_X
# WORKERS_BOT_RIGHT_X = 1030

# ARMY_TOP_LEFT_X = WORKERS_BOT_RIGHT_X
# ARMY_BOT_RIGHT_X = 1175

#NEW SPECTATOR INTERFACE
#top left, bottom right
TIME_TOP_LEFT = (0,780)
TIME_BOT_RIGHT = (90,810)

P1_TOP_LEFT_Y = 968
P1_BOT_RIGHT_Y = 1012
P2_TOP_LEFT_Y = 1036
P2_BOT_RIGHT_Y = 1080
#------
RACE_TOP_LEFT_X = 496
RACE_BOT_RIGHT_X = 530

POP_TOP_LEFT_X = 530
POP_BOT_RIGHT_X = 650

MINS_TOP_LEFT_X = POP_BOT_RIGHT_X
MINS_BOT_RIGHT_X = 795

GAS_TOP_LEFT_X = MINS_BOT_RIGHT_X
GAS_BOT_RIGHT_X = 950

WORKERS_TOP_LEFT_X = GAS_BOT_RIGHT_X
WORKERS_BOT_RIGHT_X = 1030

ARMY_TOP_LEFT_X = WORKERS_BOT_RIGHT_X + 25
ARMY_BOT_RIGHT_X = 1125

bBoxes = {"time": [TIME_TOP_LEFT,TIME_BOT_RIGHT],
		"p1_pop": [(POP_TOP_LEFT_X,P1_TOP_LEFT_Y),(POP_BOT_RIGHT_X,P1_BOT_RIGHT_Y)], 
		"p1_mins": [(MINS_TOP_LEFT_X,P1_TOP_LEFT_Y),(MINS_BOT_RIGHT_X,P1_BOT_RIGHT_Y)],
		"p1_gas": [(GAS_TOP_LEFT_X,P1_TOP_LEFT_Y),(GAS_BOT_RIGHT_X,P1_BOT_RIGHT_Y)],
		"p1_workers": [(WORKERS_TOP_LEFT_X,P1_TOP_LEFT_Y),(WORKERS_BOT_RIGHT_X,P1_BOT_RIGHT_Y)],
		"p1_army": [(ARMY_TOP_LEFT_X,P1_TOP_LEFT_Y),(ARMY_BOT_RIGHT_X,P1_BOT_RIGHT_Y)],
		"p2_pop": [(POP_TOP_LEFT_X,P2_TOP_LEFT_Y),(POP_BOT_RIGHT_X,P2_BOT_RIGHT_Y)], 
		"p2_mins": [(MINS_TOP_LEFT_X,P2_TOP_LEFT_Y),(MINS_BOT_RIGHT_X,P2_BOT_RIGHT_Y)],
		"p2_gas": [(GAS_TOP_LEFT_X,P2_TOP_LEFT_Y),(GAS_BOT_RIGHT_X,P2_BOT_RIGHT_Y)],
		"p2_workers": [(WORKERS_TOP_LEFT_X,P2_TOP_LEFT_Y),(WORKERS_BOT_RIGHT_X,P2_BOT_RIGHT_Y)],
		"p2_army": [(ARMY_TOP_LEFT_X,P2_TOP_LEFT_Y),(ARMY_BOT_RIGHT_X,P2_BOT_RIGHT_Y)]}

normLib = {'currentTime': (382.35318122912827, 295.0573702576438),
		'p1_race': (1.0,2.0),
		'p1_minerals': (318.04457209789166, 525.7697996775521), 
		'p1_gas': (247.16856277725165, 383.5032096815544), 
		'p1_foodUsed': (76.4977944474904, 55.55020035632084), 
		'p1_foodCap': (90.31232243433186, 63.04791395754466), 
		'p1_foodArmy': (35.59777949459203, 38.23587602077836), 
		'p1_foodWorkers': (39.733900712754824, 20.20304999128393), 
		'p2_race': (1.0,2.0),
		'p2_minerals': (313.3559662064497, 513.0726589221229), 
		'p2_gas': (242.93353436674477, 363.2155432504218), 
		'p2_foodUsed': (75.74890968449384, 55.05759309442984), 
		'p2_foodCap': (89.68260604097094, 62.77214140069356), 
		'p2_foodArmy': (35.41840078751932, 38.372669356857564), 
		'p2_foodWorkers': (39.16942256890794, 19.809543346619026)}

###############################################

def str2int(string):
	if not string.isdigit():
		return -1
	else:
		return int(string)

#PRE:  the image we want to process
#POST: returns a tuple of curTime, as well as the players' stats
#STATS:
#       - FoodUsed
#       - FoodCap
#       - Minerals
#       - Gas
#       - Workers
#       - Army
def getStatsFromImage(img, baseTerranImg, baseProtossImg, baseZergImg):
	stats = {}
	p1Stats = {}
	p2Stats = {}

	#Figure out the race based on an image
	p1RaceImg = np.array(img.crop((RACE_TOP_LEFT_X,P1_TOP_LEFT_Y,RACE_BOT_RIGHT_X,P1_BOT_RIGHT_Y)).convert('L'))
	p2RaceImg = np.array(img.crop((RACE_TOP_LEFT_X,P2_TOP_LEFT_Y,RACE_BOT_RIGHT_X,P2_BOT_RIGHT_Y)).convert('L'))

	p1RaceLikeness = [ssim(p1RaceImg, baseTerranImg), ssim(p1RaceImg, baseProtossImg), ssim(p1RaceImg, baseZergImg)]
	stats['p1_race'] = p1RaceLikeness.index(max(p1RaceLikeness)) + 1
	p2RaceLikeness = [ssim(p2RaceImg, baseTerranImg), ssim(p2RaceImg, baseProtossImg), ssim(p2RaceImg, baseZergImg)]
	stats['p2_race'] = p2RaceLikeness.index(max(p2RaceLikeness)) + 1

	#Game Time
	box = (bBoxes["time"][0][0],bBoxes["time"][0][1],bBoxes["time"][1][0],bBoxes["time"][1][1])
	curTimeImg = imageBotPadding(img.crop(box))
	curTime = pytesseract.image_to_string(curTimeImg,config=CONFIG).replace(" ","")
	minutes, seconds = curTime.split(":")
	if str2int(minutes) < 0 or str2int(seconds) < 0:
		stats['currentTime'] = -1
	else:
		stats['currentTime'] = str2int(minutes) * 60 + str2int(seconds)

	#-----------------------------------------------------------------

	#Player 1 population
	box = (bBoxes["p1_pop"][0][0],bBoxes["p1_pop"][0][1],bBoxes["p1_pop"][1][0],bBoxes["p1_pop"][1][1])
	p1PopImg = img.crop(box)
	popString = pytesseract.image_to_string(p1PopImg,config=CONFIG).replace(" ","")
	foodUsed,foodCap = popString.split("/")
	stats['p1_foodUsed'] = str2int(foodUsed)
	stats['p1_foodCap']  = str2int(foodCap)

	#Player 1 minerals
	box = (bBoxes["p1_mins"][0][0],bBoxes["p1_mins"][0][1],bBoxes["p1_mins"][1][0],bBoxes["p1_mins"][1][1])
	p1MinImg = img.crop(box)
	mins, income = pytesseract.image_to_string(p1MinImg,config=CONFIG).replace(" ","").split("+") #TODO: find something to do with income
	stats['p1_minerals'] = str2int(mins)

	#Player 1 gas
	box = (bBoxes["p1_gas"][0][0],bBoxes["p1_gas"][0][1],bBoxes["p1_gas"][1][0],bBoxes["p1_gas"][1][1])
	p1GasImg = img.crop(box)
	gas, income = pytesseract.image_to_string(p1GasImg,config=CONFIG).replace(" ","").split("+")
	stats['p1_gas'] = str2int(gas)

	#Player 1 Workers
	box = (bBoxes["p1_workers"][0][0],bBoxes["p1_workers"][0][1],bBoxes["p1_workers"][1][0],bBoxes["p1_workers"][1][1])
	p1WorkersImg = img.crop(box)
	stats['p1_foodWorkers'] = str2int(pytesseract.image_to_string(p1WorkersImg,config=CONFIG).replace(" ",""))

	#Player 1 Army
	# box = (bBoxes["p1_army"][0][0],bBoxes["p1_army"][0][1],bBoxes["p1_army"][1][0],bBoxes["p1_army"][1][1])
	# p1ArmyImg = img.crop(box)
	# p1Stats['Army'] = str2int(pytesseract.image_to_string(p1ArmyImg,config=CONFIG).replace(" ",""))

	stats['p1_foodArmy'] = stats['p1_foodUsed'] - stats['p1_foodWorkers']

	#-----------------------------------------------------------------

	#Player 2 population
	box = (bBoxes["p2_pop"][0][0],bBoxes["p2_pop"][0][1],bBoxes["p2_pop"][1][0],bBoxes["p2_pop"][1][1])
	p2PopImg = img.crop(box)
	popString = pytesseract.image_to_string(p2PopImg,config=CONFIG).replace(" ","")
	foodUsed,foodCap = popString.split("/")
	stats['p2_foodUsed'] = str2int(foodUsed)
	stats['p2_foodCap']  = str2int(foodCap)

	#Player 2 minerals
	box = (bBoxes["p2_mins"][0][0],bBoxes["p2_mins"][0][1],bBoxes["p2_mins"][1][0],bBoxes["p2_mins"][1][1])
	p2MinImg = img.crop(box)
	mins, income = pytesseract.image_to_string(p2MinImg,config=CONFIG).replace(" ","").split("+")
	stats['p2_minerals'] = str2int(mins)

	#Player 2 gas
	box = (bBoxes["p2_gas"][0][0],bBoxes["p2_gas"][0][1],bBoxes["p2_gas"][1][0],bBoxes["p2_gas"][1][1])
	p2GasImg = img.crop(box)
	gas, income = pytesseract.image_to_string(p2GasImg,config=CONFIG).replace(" ","").split("+")
	stats['p2_gas'] = str2int(gas)

	#Player 2 Workers
	box = (bBoxes["p2_workers"][0][0],bBoxes["p2_workers"][0][1],bBoxes["p2_workers"][1][0],bBoxes["p2_workers"][1][1])
	p2WorkersImg = img.crop(box)
	stats['p2_foodWorkers'] = str2int(pytesseract.image_to_string(p2WorkersImg,config=CONFIG).replace(" ",""))

	#TODO: fix this problem where it cant read the number
	#Player 2 army
	# box = (bBoxes["p2_army"][0][0],bBoxes["p2_army"][0][1],bBoxes["p2_army"][1][0],bBoxes["p2_army"][1][1])
	# p2ArmyImg = img.crop(box)
	# # enhancer = ImageEnhance.Contrast(p2ArmyImg.convert('LA')) #grayscale
	# # p2ArmyImg = enhancer.enhance(4.0)

	# p2ArmyImg.show()
	# pp = pytesseract.image_to_string(p2ArmyImg,config=CONFIG).replace(" ","")
	# print("PP: " + pp)
	# p2Stats['Army'] = str2int(pp)

	stats['p2_foodArmy'] = stats['p2_foodUsed'] - stats['p2_foodWorkers']


	return stats

def normalizeValues(data, normLib):
	returnData = {}
	for key in data:
		if key in normLib:
			returnData[key] = (data[key] - normLib[key][0]) / normLib[key][1]
		else:
			returData[key] = data[key]
	return returnData

def imgProcess(imgQueue, model, normLib):
	baseTerranImg = np.array(Image.open("RaceImages/terran.png").convert('L'))
	baseProtossImg = np.array(Image.open("RaceImages/protoss.png").convert('L'))
	baseZergImg = np.array(Image.open("RaceImages/zerg.png").convert('L'))

	allStats = {'currentTime': [], 'p1_race': [], 'p1_minerals': [], 'p1_gas': [], 'p1_foodUsed': [], 'p1_foodCap': [], 'p1_foodWorkers': [], 'p1_foodArmy': [],
				'p2_minerals': [], 'p2_race': [], 'p2_gas': [], 'p2_foodUsed': [], 'p2_foodCap': [], 'p2_foodWorkers': [], 'p2_foodArmy': []}

	running = True
	while running:
		try:
			element = imgQueue.get()
			if element == None:  # Signal to print and exit NOW!
				running = False
				break

			img = element
			stats = getStatsFromImage(img, baseTerranImg, baseProtossImg, baseZergImg)
			print("Time: " + str(stats['currentTime']))
			print(stats)

			#TODO: make this work on images
			# stats['p1_race'].append(0.0)
			# stats['p2_race'].append(0.5)

			normStats = normalizeValues(stats, normLib)

			dfStats = pd.DataFrame(normStats.items())
			guess = model.predict(dfStats)

			# value = 0
			# if team2WinPred > team1WinPred:
			# 	# TODO: decide if this subtraction is needed. The 
			# 	# Thought behind it is if both are equally likely 
			# 	# to be true, it should be a draw.
			# 	value = -1 * (team2WinPred - team1WinPred)
			# else:
			# 	value = (team1WinPred - team2WinPred)
			#xs = np.append(xs,i)
			# ys = np.append(ys,value)

			ys = np.append(ys,guess)

			# Limit x and y lists to the last 50 elements
			#xs = xs[-50:]
			ys = ys[-50:]

			lines = live_plotter(xs,ys,lines)

			for key in stats:
				allStats[key].append(stats[key])

		except Exception as e: #Empty queue
			pass

##################################

#modelPath = "Models/ResNet18_BinaryClassification_FullVision/"
#modeFileName = "model_00_0.pth"
#modelPath = FLAGS.model

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
		#TODO: check pixels for mineral and gas
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

#MINIMAP CODE#####################################
# Load model class
#model = torch.load(modelPath+modeFileName)
#model = torch.load(modelPath)
#model.eval()
##################################################

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

#MINIMAP CODE#####################################
# normalize = transforms.Normalize(
# 	mean=[0.1213, 0.1105, 0.1275],
# 	std=[0.1201, 0.1079, 0.1408]
# )

# preprocess = transforms.Compose([
#    transforms.ToTensor(),
#    normalize
# ])
###################################################

# Create figure for plotting
plt.style.use('ggplot')
size = 50
xs = np.linspace(0,1,size+1)[0:-1]
ys = np.linspace(0,0,size+1)[0:-1]
lines = []
########

model = load(FLAGS.model)

imgQueue = multiprocessing.Queue()
imgProcessThread = threading.Thread(target=imgProcess, args=(imgQueue,model,normLib,))
imgProcessThread.start()
fps = 60
timeInterval = 5
curFrame = fps * timeInterval #start with a prediction

########


i = 0
data = bytearray()
while True:
	try:
		#print("Gathering Stream Data...")
		fd = stream.open()
		data = fd.read(400024)

		#DOWNLOAD A STREAM#################################
		# while i < 10:
		# 	data = data + fd.read(4000024)
		# 	i += 1
		# fd.close()
		# fname = 'stream2.mp4'
		# f = open(fname, 'wb')
		# f.write(data)
		# f.close()
		# exit()
		####################################################

		if curFrame % (60 * timeInterval) != 0:
			curFrame += 1
			continue
		else:
			curFrame = 1

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

			imgQueue.put(img)

			#MINIMAP CODE#####################################
			#img.save(destinationFolder+frameImage) #save the full screen image
			# croppedImage   = img.crop(streamImageBox)
			# paddedImage    = createPaddedImage(croppedImage, imageResize)

			# if FLAGS.save_images == True:
			# 	paddedImage.save(destinationFolder+"resized"+frameImage)
			# imgTensor = preprocess(paddedImage)
			# imgTensor.unsqueeze_(0)

			# output = model(imgTensor.cuda())
			# print("TENSOR: " + str(output))
			# print("OUTPUT: " + str(output[0][0].item()))

			# Add x and y to lists
			#xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
			# team1WinPred = output[0][0].item()
			# team2WinPred = output[0][1].item()
			##################################################
			

			# if value < 0:
			# 	newsegm, = ax.plot(xs, ys, color='red')
			# else:
			# 	newsegm, = ax.plot(xs, ys, color='blue')

			#else:
			#print("NOT GAME")
		else:
			print("Imaged Failed to Captured.") 

		i += 1
	except (KeyboardInterrupt, SystemExit):
		capture.release()
		imgQueue.put(None) # Tell the data_queue to exit.
		imgProcessThread.join()
		break
	except Exception as e:
		print(e)


####################################

#ani = animation.FuncAnimation(fig, animate, frames=50, fargs=(stream, xs, ys), interval=1000)
# ani = animation.FuncAnimation(fig, animate, frames=50, interval=1000)
# plt.show()


#capture.release()
	# #TODO: figure out a way to see if you won by taking pictures of the location of the victory message