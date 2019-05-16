import os
import shutil
from os import walk

inputFolder  = 'WCS_Austin_2018'
outputFolder = 'Games_Austin'

if not(os.path.isdir(inputFolder)):
	print("'" + inputFolder + "' Replay Folder does not exist.")
	exit()

if not(os.path.isdir(outputFolder)):
	os.mkdir(outputFolder.strip("/"))

r = []
f = []
for root, dirs, files in os.walk(inputFolder, topdown=False):
	for name in files:
		r.append(root+"/")
		f.append(name)

for i in range(1,len(f)):
	print(r[i]+f[i])
	shutil.copy2(r[i]+f[i],outputFolder+"/"+str(i)+".SC2Replay")

print("Finished saving games.")