from shutil import copyfile
import os

imagePath = "resizedFullVisionImages/"
classOnePath = "resizedFullVisionImages/001/"
classTwoPath = "resizedFullVisionImages/002/"

LastPath = "Last/"

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