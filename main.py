##########################################
#       RESIZE IMAGES WITH PADDING       #
##########################################

# RESIZE IMAGES WITH PADDING
from PIL import Image, ImageOps

from os import listdir
from os.path import isfile, join
from shutil import rmtree

print("Resizing images to be the correct size with padding...")

desiredImageSize = 256
prePaddedImageFilePath = "FullVisionImages/"
paddedImageFilePath = "resizedFullVisionImages/"

if not(os.path.isdir(paddedImageFilePath)):
	os.mkdir(paddedImageFilePath)
else:
	#delete files and create new ones
	shutil.rmtree(paddedImageFilePath)
	os.mkdir(paddedImageFilePath)

imageFilesPrePadding = [f for f in listdir(prePaddedImageFilePath.strip('/')) if isfile(join(prePaddedImageFilePath.strip('/'), f))]

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

import math

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

from os import rename

imagePath = "resizedFullVisionImages/"
classOnePath = "resizedFullVisionImages/001/"
classTwoPath = "resizedFullVisionImages/002/"

if not(os.path.isdir(classOnePath)):
	os.mkdir(classOnePath)
else:
	#delete files and create new ones
	shutil.rmtree(classOnePath)
	os.mkdir(classOnePath)

if not(os.path.isdir(classTwoPath)):
	os.mkdir(classTwoPath)
else:
	#delete files and create new ones
	shutil.rmtree(classTwoPath)
	os.mkdir(classTwoPath)

datasetFile = "datasetFullVision.csv"

dataFile = open(datasetFile,"r")

lines = dataFile.readlines()
dataFile.close()

for line in lines:
	stripLine = line.strip("\n").split(",")
	#file index 0
	#outcome 8

	if stripLine[8] == "1":
		if os.path.isfile(imagePath + stripLine[0]):
			os.rename(imagePath + stripLine[0], classOnePath + stripLine[0])
	else:
		if os.path.isfile(imagePath + stripLine[0]):
			os.rename(imagePath + stripLine[0], classTwoPath + stripLine[0])

