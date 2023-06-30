# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:26:26 2023

@author: Maxence
"""

from flask import Flask,request,jsonify
from flask_cors import CORS
import librosa
import os
app = Flask(__name__)
CORS(app)


@app.route('/dereverberation')

def dereverberation():
    os.system("python demo.py --audiofilelist myaudiofiles.scp")
    beat_tracking("tempderev.wav")
    

@app.route('/send_wav', methods=['POST'])

def save_temp_file():
    wav_data = request.data
    #save the file
    temp_filename = "temp.wav"
    with open(temp_filename,'wb') as file:
        file.write(wav_data)

@app.route('/beatTrack', methods=['POST'])

def beat_tracking(fileworkname = "temp.wav"):
    y,sr = librosa.load(fileworkname)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats,sr=sr).tolist()
    
    response_data = "{'beat_times':" + str(beat_times)+" ,'BPM' : "+str(tempo)+"}"
    return response_data

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)