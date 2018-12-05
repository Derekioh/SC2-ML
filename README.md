# SC2 ML

Dependencies
pip install win32gui
pip install pywin32
pip install Image

Dataset
RACE
----
terran  = 1
zerg    = 2
protoss = 3

GAME DURATION
-------------
In seconds

OUTCOME
-------
Win  = 1
Loss = 2

Example Game:

GAME,0
1,3,286,1              #player id, player race, palyer apm, win/loss
2,1,383,2              #player id, player race, palyer apm, win/loss
9320,416.1004638671875 #time steps, time in seconds of whole game


STEPS TO GET JUPYTER WORKING

ON LILOU
jupyter notebook --no-browser --port=8080

On LINUX Terminal on PC
ssh -N -L 8080:localhost:8080 dtlafever@lilou.seas.gwu.edu

ON LILOU
Copy the url that is provided or go to http://localhost:8080/