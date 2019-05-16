try:
    from PIL import Image
    from PIL import ImageOps
    from PIL import ImageEnhance
except ImportError:
    import Image
import pytesseract
import cv2
import numpy as np
import threading
import multiprocessing
from skimage.measure import compare_ssim as ssim #Determine the similarity of images

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

#CONFIG = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789-.:+/'
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

####################################################################

def imageBotPadding(im, padding=0):

	width, height = im.size
	returnIm = np.array(im)
	if padding != 0:
		x = np.uint8(np.array([[[0,0,0] for i in range(width)] for j in range(padding)]))
		# print(x.shape)
		# print(returnIm.shape)
		#print(returnIm)
		returnIm = np.append(returnIm,x,axis=0)

	returnIm = Image.fromarray(returnIm)

	return returnIm

def str2int(string):
	if not string.isdigit():
		return -1
	else:
		return int(string)

####################################################################

#PRE:  the image we want to process
#POST: returns a tuple of curTime, as well as the players' stats
#STATS:
#       - FoodUsed
#       - FoodCap
#       - Minerals
#       - Gas
#       - Workers
#       - Army
def getStatsFromImage(img):
	p1Stats = {}
	p2Stats = {}

	#Game Time
	box = (bBoxes["time"][0][0],bBoxes["time"][0][1],bBoxes["time"][1][0],bBoxes["time"][1][1])
	curTimeImg = imageBotPadding(img.crop(box))
	curTime = pytesseract.image_to_string(curTimeImg,config=CONFIG).replace(" ","")
	minutes, seconds = curTime.split(":")
	if str2int(minutes) < 0 or str2int(seconds) < 0:
		curTime = -1
	else:
		curTime = str2int(minutes) * 60 + str2int(seconds)

	#-----------------------------------------------------------------

	#Player 1 population
	box = (bBoxes["p1_pop"][0][0],bBoxes["p1_pop"][0][1],bBoxes["p1_pop"][1][0],bBoxes["p1_pop"][1][1])
	p1PopImg = img.crop(box)
	popString = pytesseract.image_to_string(p1PopImg,config=CONFIG).replace(" ","")
	foodUsed,foodCap = popString.split("/")
	p1Stats['FoodUsed'] = str2int(foodUsed)
	p1Stats['FoodCap']  = str2int(foodCap)

	#Player 1 minerals
	box = (bBoxes["p1_mins"][0][0],bBoxes["p1_mins"][0][1],bBoxes["p1_mins"][1][0],bBoxes["p1_mins"][1][1])
	p1MinImg = img.crop(box)
	mins, income = pytesseract.image_to_string(p1MinImg,config=CONFIG).replace(" ","").split("+") #TODO: find something to do with income
	p1Stats['Minerals'] = str2int(mins)

	#Player 1 gas
	box = (bBoxes["p1_gas"][0][0],bBoxes["p1_gas"][0][1],bBoxes["p1_gas"][1][0],bBoxes["p1_gas"][1][1])
	p1GasImg = img.crop(box)
	gas, income = pytesseract.image_to_string(p1GasImg,config=CONFIG).replace(" ","").split("+")
	p1Stats['Gas'] = str2int(gas)

	#Player 1 Workers
	box = (bBoxes["p1_workers"][0][0],bBoxes["p1_workers"][0][1],bBoxes["p1_workers"][1][0],bBoxes["p1_workers"][1][1])
	p1WorkersImg = img.crop(box)
	p1Stats['Workers'] = str2int(pytesseract.image_to_string(p1WorkersImg,config=CONFIG).replace(" ",""))

	#Player 1 Army
	# box = (bBoxes["p1_army"][0][0],bBoxes["p1_army"][0][1],bBoxes["p1_army"][1][0],bBoxes["p1_army"][1][1])
	# p1ArmyImg = img.crop(box)
	# p1Stats['Army'] = str2int(pytesseract.image_to_string(p1ArmyImg,config=CONFIG).replace(" ",""))

	p1Stats['Army'] = p1Stats['FoodUsed'] - p1Stats['Workers']

	#-----------------------------------------------------------------

	#Player 2 population
	box = (bBoxes["p2_pop"][0][0],bBoxes["p2_pop"][0][1],bBoxes["p2_pop"][1][0],bBoxes["p2_pop"][1][1])
	p2PopImg = img.crop(box)
	popString = pytesseract.image_to_string(p2PopImg,config=CONFIG).replace(" ","")
	foodUsed,foodCap = popString.split("/")
	p2Stats['FoodUsed'] = str2int(foodUsed)
	p2Stats['FoodCap']  = str2int(foodCap)

	#Player 2 minerals
	box = (bBoxes["p2_mins"][0][0],bBoxes["p2_mins"][0][1],bBoxes["p2_mins"][1][0],bBoxes["p2_mins"][1][1])
	p2MinImg = img.crop(box)
	mins, income = pytesseract.image_to_string(p2MinImg,config=CONFIG).replace(" ","").split("+")
	p2Stats['Minerals'] = str2int(mins)

	#Player 2 gas
	box = (bBoxes["p2_gas"][0][0],bBoxes["p2_gas"][0][1],bBoxes["p2_gas"][1][0],bBoxes["p2_gas"][1][1])
	p2GasImg = img.crop(box)
	gas, income = pytesseract.image_to_string(p2GasImg,config=CONFIG).replace(" ","").split("+")
	p2Stats['Gas'] = str2int(gas)

	#Player 2 Workers
	box = (bBoxes["p2_workers"][0][0],bBoxes["p2_workers"][0][1],bBoxes["p2_workers"][1][0],bBoxes["p2_workers"][1][1])
	p2WorkersImg = img.crop(box)
	p2Stats['Workers'] = str2int(pytesseract.image_to_string(p2WorkersImg,config=CONFIG).replace(" ",""))

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

	p2Stats['Army'] = p2Stats['FoodUsed'] - p2Stats['Workers']


	return curTime, p1Stats, p2Stats

def imgProcess(imgQueue):
	CurTime = []
	p1 = {'FoodUsed': [], 'FoodCap': [], 'Minerals': [], 'Gas': [], 'Workers': [], 'Army': []}
	p2 = {'FoodUsed': [], 'FoodCap': [], 'Minerals': [], 'Gas': [], 'Workers': [], 'Army': []}

	running = True
	while running:
		try:
			element = imgQueue.get()
			if element == None:  # Signal to print and exit NOW!
				running = False
				break

			img = element
			curTime, p1Stats, p2Stats = getStatsFromImage(img)
			print("Time: " + str(curTime))
			print(p1Stats)
			print(p2Stats)

			CurTime.append(curTime)
			for key in p1:
				p1[key].append(p1Stats[key])
				p2[key].append(p2Stats[key])

		except Exception as e: #Empty queue
			pass

####################################################################  

#cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_crop)

imgQueue = multiprocessing.Queue()
imgProcessThread = threading.Thread(target=imgProcess, args=(imgQueue,))
imgProcessThread.start()

capture = cv2.VideoCapture("stream2.mp4")

curFrame = 1
fps = 60
timeInterval = 5

DEBUG = 1

while(capture.isOpened()):
	ret, frame = capture.read()

	#print(curFrame)
	if curFrame % (60 * timeInterval) != 0:
		curFrame += 1
		continue
	else:
		curFrame = 1

	if ret == True:

		img = Image.fromarray(frame)
		#convert from BGR to RGB
		b,g,r = img.split()
		img = Image.merge("RGB", (r,g,b))

		if not DEBUG:
			imgQueue.put(img)

		if DEBUG:
			baseTerranImg = np.array(Image.open("RaceImages/terran.png").convert('L'))
			baseProtossImg = np.array(Image.open("RaceImages/protoss.png").convert('L'))
			baseZergImg = np.array(Image.open("RaceImages/zerg.png").convert('L'))
			p1RaceImg = np.array(img.crop((RACE_TOP_LEFT_X,P1_TOP_LEFT_Y,RACE_BOT_RIGHT_X,P1_BOT_RIGHT_Y)).convert('L'))
			p2RaceImg = np.array(img.crop((RACE_TOP_LEFT_X,P2_TOP_LEFT_Y,RACE_BOT_RIGHT_X,P2_BOT_RIGHT_Y)).convert('L'))

			print("P1: " + "T=" + str(ssim(p1RaceImg, baseTerranImg)) + " P=" + str(ssim(p1RaceImg, baseProtossImg)) + " Z=" + str(ssim(p1RaceImg, baseZergImg)))
			print("P2: " + "T=" + str(ssim(p2RaceImg, baseTerranImg)) + " P=" + str(ssim(p2RaceImg, baseProtossImg)) + " Z=" + str(ssim(p2RaceImg, baseZergImg)))

			cv2.rectangle(frame,(496,968),(530,1012),(0,255,0),3)
			cv2.rectangle(frame,(496,1036),(530,1080),(0,255,0),3)

			# for key, value in bBoxes.items():
			# 	cv2.rectangle(frame,value[0],value[1],(0,255,0),3)
			print(getStatsFromImage(img))
			cv2.imshow("image", frame)
			cv2.waitKey(0)
			break
	else:
		#TODO fix this so if we have droppped frames it won't die
		break

capture.release()
imgQueue.put(None) # Tell the data_queue to exit.
imgProcessThread.join()
cv2.destroyAllWindows()