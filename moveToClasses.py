import os

imagePath = "resizedFullVisionImages/"
classOnePath = "resizedFullVisionImages/001/"
classTwoPath = "resizedFullVisionImages/002/"

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