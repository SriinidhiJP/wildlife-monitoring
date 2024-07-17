from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import numpy as np
import argparse
import pickle
import cv2
import os
import time
from keras.models import load_model
from collections import deque
from twilio.rest import Client
from playsound import playsound
import pygame
import winsound
sms_has_sent=False
audio_has_played=False
account_sid ="ACd844a................." 
auth_token ="ea65d................." 
client = Client(account_sid, auth_token)
pygame.init()
detected_sound = pygame.mixer.Sound (r"C:\Users\panit\Desktop\py project\dog- barking-70772.mp3”)
def sms():
global sms_has_sent
if sms_has_sent:
return
sum_has_sent = True
return client.api.account.messages.create(
to="+91 9000000000",
from_="+16813846818",
body="WILD BOAR DETECTED! PROTECT YOUR CROPS!!")
def audio():
global audio_has_played
if audio_has_played:
return
audio_has_played=True
return None
def print_results(video, limit=None):
if not os.path.exists('output'):
os.mkdir('output')
print("Loading model ...")
model = load_model(r"C:\Users\panit\Desktop\py project\model.h5")
Q = deque(maxlen=128)
vs = cv2.VideoCapture(video)
writer = None
(W, H) = (None, None)
count = 0
while True:
(grabbed, frame) = vs.read()
if not grabbed:
break
if W is None or H is None:
(H, W) = frame.shape[:2]
output = frame.copy()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (150,150)).astype("float32")
frame = frame.reshape(150,150, 3) / 255
preds = model.predict(np.expand_dims(frame, axis=0))[0]
Q.append(preds)
results = np.array(Q).mean(axis=0)
i = (preds > 0.50)[0]
label = i
print(label)
print(preds)
text_color = (0, 255, 0) # default : green
if label: # Violence prob
text_color = (0, 0, 255) # red
text = "Class: {}".format(label)
FONT = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3)
cv2.putText(output, "WILD BOAR DETECTED", (35, 150), FONT,1.25, text_color, 3)
cv2.imshow("Output", output)
detected_sound.play()
sms()
else:
text_color = (0, 255, 0)
text = "Class: {}".format(label)
FONT = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3)
cv2.putText(output, "WILD BOAR NOT DETECTED", (35, 150), FONT,1.25, text_color, 3)
cv2.imshow("Output", output
cv2.imshow("Output", output)
key = cv2.waitKey(1) & 0xFF
if key == ord("q"):
break
print("[INFO] cleaning up...")
V_path = r"C:\Users\panit\Desktop\py project\pos2.mp4" print_results(V_path)
