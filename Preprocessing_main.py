from PIL import Image, ImageOps

import os
from os.path import isfile, join
from shutil import rmtree
from shutil import copyfile
import math

##########################################
#       RESIZE IMAGES WITH PADDING       #
##########################################

print("Resizing images to be the correct size with padding...")

desiredImageSize = 256
prePaddedImageFilePath = "FullVisionImages/"
paddedImageFilePath = "resizedFullVisionImages/"

if not(os.path.isdir(paddedImageFilePath)):
	os.mkdir(paddedImageFilePath.strip("/"))
else:
	#delete files and create new ones
	rmtree(paddedImageFilePath)
	os.mkdir(paddedImageFilePath.strip("/"))

imageFilesPrePadding = [f for f in os.listdir(prePaddedImageFilePath.strip('/')) if isfile(join(prePaddedImageFilePath.strip('/'), f))]

for file in imageFilesPrePadding:

	im = Image.open(prePaddedImageFilePath+file)
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

	#new_im.show()
	new_im.save(paddedImageFilePath+file)

print("Resizing complete.")

##########################################
#       PARSE DATA AND CREATE A CSV      #
##########################################

print("Parsing data file and creating a CSV from it...")

dataFilename = "dataFullVision.txt"
datasetFilename = "datasetFullVision.csv"

dataFile = open(dataFilename,'r')
datasetFile = open(datasetFilename,'w')

#write header for the csv file
datasetFile.write("file,current_time,race0,apm0,race1,apm1,map,total_time,outcome\n")

lines = dataFile.readlines()
dataFile.close()

dataRow = []
for line in lines:
	splitLine = line.strip("\n").split(",")
	if splitLine[0] == "GAME":
		dataRow = splitLine[2:]
	else:
		#only write if the filename isn't ERROR
		if splitLine[3] != "ERROR":
			datasetFile.write(splitLine[3] + "," + splitLine[2] + ",")
			for ele in dataRow[:-2]:
				datasetFile.write(ele + ",")
			datasetFile.write(str(math.floor(float(dataRow[-2]))) + ",")
			datasetFile.write(dataRow[-1] + "\n")

datasetFile.close()

print("Parsing Complete.")

##########################################
#       MOVE IMAGES TO CLASS FOLDERS     #
##########################################

# Time when it is undetermined who will win
DRAW_TIME = 60

imagePath = "resizedFullVisionImages/"
classZeroPath = imagePath + "000/"
classOnePath  = imagePath + "001/"
classTwoPath  = imagePath + "002/"

if not(os.path.isdir(classZeroPath)):
	os.mkdir(classZeroPath.strip("/"))
else:
	#delete files and create new ones
	shutil.rmtree(classZeroPath)
	os.mkdir(classZeroPath.strip("/"))

if not(os.path.isdir(classOnePath)):
	os.mkdir(classOnePath.strip("/"))
else:
	#delete files and create new ones
	shutil.rmtree(classOnePath)
	os.mkdir(classOnePath.strip("/"))

if not(os.path.isdir(classTwoPath)):
	os.mkdir(classTwoPath.strip("/"))
else:
	#delete files and create new ones
	shutil.rmtree(classTwoPath)
	os.mkdir(classTwoPath.strip("/"))

datasetFile = "datasetFullVision.csv"

dataFile = open(datasetFile,"r")

lines = dataFile.readlines()
dataFile.close()

for line in lines[1:]:
	stripLine = line.strip("\n").split(",")
	#file index 0
	#outcome 8
	#current_time 1

	# if below 60 seconds, consider the game a draw
	#print(stripLine)
	if int(stripLine[1]) <= DRAW_TIME:
		if os.path.isfile(imagePath + stripLine[0]):
				os.rename(imagePath + stripLine[0], classZeroPath + stripLine[0])
	else:
		if stripLine[8] == "1":
			if os.path.isfile(imagePath + stripLine[0]):
				os.rename(imagePath + stripLine[0], classOnePath + stripLine[0])
		else:
			if os.path.isfile(imagePath + stripLine[0]):
				os.rename(imagePath + stripLine[0], classTwoPath + stripLine[0])

##########################################
#       COPY LAST 30 SEC IMAGES to FOLDER#
##########################################

imagePath = "resizedFullVisionImages/"
classZeroPath = imagePath + "000/"
classOnePath  = imagePath + "001/"
classTwoPath  = imagePath + "002/"

LastPath = "Last/"

if not(os.path.isdir(classZeroPath+LastPath)):
	os.mkdir(classZeroPath+LastPath.strip("/"))
else:
	#delete files and create new ones
	shutil.rmtree(classZeroPath+LastPath)
	os.mkdir(classZeroPath+LastPath.strip("/"))

if not(os.path.isdir(classOnePath+LastPath)):
	os.mkdir(classOnePath+LastPath.strip("/"))
else:
	#delete files and create new ones
	shutil.rmtree(classOnePath+LastPath)
	os.mkdir(classOnePath+LastPath.strip("/"))

if not(os.path.isdir(classTwoPath+LastPath)):
	os.mkdir(classTwoPath+LastPath.strip("/"))
else:
	#delete files and create new ones
	shutil.rmtree(classTwoPath+LastPath)
	os.mkdir(classTwoPath+LastPath.strip("/"))

datasetFile = "datasetFullVision.csv"

dataFile = open(datasetFile,"r")

lines = dataFile.readlines()
dataFile.close()

for line in lines[1:]:
	stripLine = line.strip("\n").split(",")
	#file index 0
	#outcome 8

	if (int(stripLine[7]) - int(stripLine[1])) <= 30:
		if stripLine[8] == "1":
			if os.path.isfile(classOnePath + stripLine[0]):
				copyfile(classOnePath + stripLine[0], classOnePath + LastPath + stripLine[0])
		else:
			if os.path.isfile(classTwoPath + stripLine[0]):
				copyfile(classTwoPath + stripLine[0], classTwoPath + LastPath + stripLine[0])