import math

dataFilename = "dataFullVision.txt"
datasetFilename = "datasetFullVision.csv"

dataFile = open(dataFilename,'r')
datasetFile = open(datasetFilename,'a')

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