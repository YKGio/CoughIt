import mido
from pydub import AudioSegment
import numpy as np
import soundfile
import librosa
import os
import random


def midi2wav():
    ### LOAD AND PARSE ###
    print('Parsing MIDI files...')
    
    cough_paths = []
    for root, dirs, files in os.walk('./audios/'):
        for f in files:
            cough_paths.append(os.path.join(root, f))
    
    coughs = []  
    for path in cough_paths:     
        cough, sr = librosa.load(path, sr=48000)
        coughs.append(cough)
        
    
    mid = mido.MidiFile('./MIDI/generated.mid')
    track = mid.tracks[1][1:-1]

    note_seq = []
    for note in track:
        if note.velocity == 80:
            note_seq.append(note.note)
           
    ### PITCH SHIFT ###
    print('Converting to WAV...') 
    out = np.array(0)      
    for note in note_seq:
        bias = note - 69
        cough = random.choice(coughs)
        y_shifted = librosa.effects.pitch_shift(cough, 48000, n_steps=bias)
        out = np.append(out, y_shifted)


    soundfile.write('out.wav', out, 48000)

    song = AudioSegment.from_wav("out.wav")
    song = song + 10

    song.export("out_loud.wav", "wav")


    print('Convert complete!')
    