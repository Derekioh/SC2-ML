# SC ML Project

## About

The Goal of this project was predict who would win in a game of Starcraft II. Using ResNet-18, I trained the network on map images taken throughout matches between professional players. With this network, I achieved an accuracy of 64%. A more detailed writeup of the work can be read in the *final_paper.pdf*.

The dataset of replays (included) is taken from the WCS Valencia 2018 tournament (406 replays). My code is broken up into two main parts:

1. Data collection and processing (Ran in Windows)
2. training and testing

As such, I will break down requirements and usage accordingly. Note that data colleciton and processing was done in windows to make use of the Starcraft II game replay feature. While there has been Starcraft II environmments that are tested in Linux, I have not personally used them.

For more details on results

## Requirements

### Data Collection and Processing

Python 3 requirements
```
pip install shutil
pip install mss
pip install pysc2
```
Additionally, [Starcraft II](https://starcraft2.com/en-us/) will need to be downloaded and installed.

### Training and Testing

Anaconda 3.5.1
```
PyTorch 0.4.1
```

### Live Processing on a Twitch Stream

```
pip install absl-py
pip install Pillow
pip install livestreamer
pip install opencv-python
pip install pytesseract (version 3.X, NOT 4.X due to a bug in tesseract)
<!-- pip install oauthlib -->
pip install joblib
pip install scikit-image
pip install sklearn
PyTorch 0.4.1
```

To use pytesseract, make sure you install [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract) is installed and has the appropriate PATH variables set.

## Usage

### Data Collection and Processing

While the replays to the tournament are provided in this repo, you will need to generate the map images from the replays. This means running every game and taking periodic screenshots of the map image to be later fed into our network. This capture process can be done by calling this script:

```
python obtainMinimapData.py --replays "./WCS_Valencia_2018/2018 WCS Valencia/Day 1 - RO80/Group Stage 1/Group A/Neeb vs. Okaneby" --disable_fog True --time_interval 5
```

The flags that are important here are:
 - replays = the Path to a directory of replays
 - disable_fog: whether to show replays from one player's perspective or from both
  - NOTE: setting disable_fog to false will double the number of images as each game will be run twice (once for each player's perspective)
 - time_interval: how many seconds to wait in between taking a screenshot.

FAQ: *When running this script, my saved images are all the same image after some amount of replays are processed. What do I do?*
Answer: Make sure that your screensaver is turned off and your monitor does not go to sleep!

Once this data collection is done, we can preprocess the images to be the right size for our network and will the correct labels in a CSV file:

```
python Preprocessing_main.py
```

### Training and Testing

Once we have collected the data and created our CSV files, we can train our network. Simply call

```
python ML_main.py
```

### Live Processing on a Twitch Stream

```
python liveProcessing.py --stream "esl_sc2" --model path_to_model
```

### MISC

Ladder maps for a larger dataset

https://github.com/Blizzard/s2client-proto#downloads