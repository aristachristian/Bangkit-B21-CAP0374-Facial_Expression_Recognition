from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import cv2
import librosa
from pydub import AudioSegment
from pydub.playback import play
import moviepy.editor
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tqdm import tqdm

app = Flask(__name__)

ferPath = "./ferModel"
serPath = "./serModel"

faceExpression = ["angry", "happy", "sad", "neutral"]
speechEmotion = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

testsetPath = "./uploads"
audioPath = "./extracted audio"

ferModel = load_model(ferPath)
serModel = load_model(serPath)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img, model):
    x = image.img_to_array(img)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

def prepareSound(filePath) :
    sw, sr = librosa.load(filePath, sr=22050)
    mfcc = librosa.feature.mfcc(sw, sr=sr, n_fft=2048, n_mfcc=13, hop_length=512)
    mfcc = mfcc.T
    return np.resize(mfcc, (1, 150, 13))

def extractAudioFromVideo(fName) :
    video = moviepy.editor.VideoFileClip(os.path.join(testsetPath, fName + ".mp4"))
    video.audio.write_audiofile(os.path.join(audioPath, fName + ".wav"))

def getFeedback(face, speech) :
    feedback = ""
    feedback += "Feedback For your Video :"
    feedback += " Facial Expression :"
    if face == 0 : #angry
        feedback += " In the video, you look angry and rude. You need to relax and smile more in an interview."
    elif face == 1 : #happy
        feedback += " Great, you look calm and happy in the video. You're ready for the interview."
    elif face == 2 : #sad
        feedback += " You look sad in the video. You need to smile more while answering an interview."
    elif face == 3 : #neutral
        feedback += " Your expression is too flat. Be more enthusiast in answering the interview."
        
    feedback += " Speech Tone :"
    if speech == 0 : #neutral
        feedback += " Your tone is too flat. Raise your voice a little bit so you sound more enthusiastic."
    if speech == 1 : #calm
        feedback += " Great, you sound calm in answering the interview. You're ready for the interview."
    if speech == 2 : #happy
        feedback += " Great, you sound excited in answering the interview. You're ready for the interview."
    if speech == 3 : #sad
        feedback += " You sound sad and your voice is too quite. Increase your speech volume so your interviewer can hear you better."
    if speech == 4 : #angry
        feedback += " You sound angry in answering. You need to lower your voice."
    if speech == 5 : #fearful
        feedback += " You sound nervous in the video. You need to relax and answer more calmly."
    if speech == 6 : #disgust
        feedback += " You sound disgusted and rude in answering the interview. You need to answer more calmly."
    if speech == 7 : #surprised
        feedback += " You sound surprised in the video. You need to relax and answer more calmly."
    
    return feedback

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, './uploads', secure_filename(f.filename))
        f.save(file_path)
        
        fName = file_path.split("/")[-1].split(".")[0]
        #print("File Name : {}".format(fName))
        
        video = cv2.VideoCapture(file_path)
        faceCascade = cv2.CascadeClassifier("Face-Detection-OpenCV-master/data/haarcascade_frontalface_alt.xml")
        totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        count = [0, 0, 0, 0]

        for i in tqdm(range(0, totalFrames, 5)) :
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img = video.read()
            faces = np.array(faceCascade.detectMultiScale(img, 1.1, 4))

            if not ret:
                raise Exception("Problem reading frame", i, " from video")
            
            if faces.any() :
                x, y, w, h = faces[0]

                imgFace = img[y:y + h, x:x + w]
                imgFace = cv2.cvtColor(imgFace, cv2.COLOR_BGR2RGB)
                imgFace = cv2.resize(imgFace, (48, 48))
                imgFace = np.array([imgFace])

                ferPred = ferModel.predict(imgFace)
                
                count[np.argmax(ferPred)] += 1

        if fName + ".wav" not in os.listdir(audioPath) :
            extractAudioFromVideo(fName)

        x = prepareSound(os.path.join(audioPath, fName + ".wav"))
        
        serPred = serModel.predict(x)

        result = getFeedback(np.argmax(count), np.argmax(serPred[0]))
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
