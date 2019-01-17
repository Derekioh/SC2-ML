import math

totalNumGames = 406

avgAPM = 0
avgGameLength = 0

# terran  = 1
# zerg    = 2
# protoss = 3
TerranCount  = 0
ZergCount    = 0
ProtossCount = 0

#-------

dataFile = open("out.txt","r")

lines = dataFile.readlines()
dataFile.close()

for line in lines:
	splitLine = line.strip("\n").split(",")
	if splitLine[0] == "GAME":
		if splitLine[2] == "1":
			TerranCount += 1
		elif splitLine[2] == "2":
			ZergCount += 1
		elif splitLine[2] == "3":
			ProtossCount += 1

		avgAPM += int(splitLine[3])

		if splitLine[4] == "1":
			TerranCount += 1
		elif splitLine[4] == "2":
			ZergCount += 1
		elif splitLine[4] == "3":
			ProtossCount += 1

		avgAPM += int(splitLine[5])

		avgGameLength += math.floor(float(splitLine[7]))


avgAPM = round(avgAPM / (totalNumGames * 2))
avgGameLength = round(avgGameLength / totalNumGames)

print("Total Num Games: " + str(totalNumGames))
print("Average APM: " + str(avgAPM))
print("Average Game Length: " + str(avgGameLength))
print("Terrans: " + str(TerranCount))
print("Zerg: " + str(ZergCount))
print("Protoss: " + str(ProtossCount))