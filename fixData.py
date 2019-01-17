dataFilename = "data.txt"
saveFilename = "out.txt"

dataFile = open(dataFilename,'r')

lines = dataFile.readlines()
dataFile.close()

saveFile = open(saveFilename,'a')

dataRow = []
for line in lines[1346:]:
	splitLine = line.strip("\n").split(",")
	if splitLine[0] == "GAME":
		outcome = 1
		if splitLine[8][0] == "2":
			outcome = 2
		newRow = [
				splitLine[0], #GAME
				splitLine[1], #replay number
				splitLine[3], #player 1 race
				splitLine[4], #player 1 apm
				splitLine[6], #player 2 race
				splitLine[7], #player 2 apm
				splitLine[8][1:], #map
				splitLine[10], #total time
				str(outcome)
				]
		for word in newRow[:-1]:
			saveFile.write(word + ",")
		saveFile.write(newRow[-1] + "\n")
	else:
		saveFile.write(line)

saveFile.close()