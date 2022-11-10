import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd
import soundfile as sf
import time

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input

if __name__ == "__main__":

    ################### SETTINGS ###################
    class_labels=True
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = params.SAMPLE_RATE
    WIN_SIZE_SEC = 0.975
    CHUNK = int(WIN_SIZE_SEC * RATE)
    RECORD_SECONDS = 20

    print(sd.query_devices())
    MIC = None

    #################### MODEL #####################
    
    model = YAMNet(weights='keras_yamnet/yamnet.h5')
    yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')

    #################### STREAM ####################
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT,
                        input_device_index=MIC,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # Waveform
        wave_data = np.fromstring(stream.read(CHUNK), dtype=np.float32)
        preprocessed_data = preprocess_input(wave_data, RATE)
        prediction = model.predict(np.expand_dims(preprocessed_data,0))[0]
        
        if max(prediction) == prediction[42]:   # #42 cough
            print('cough')
            t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            sf.write('./recorded_cough/' + t +'_cough.wav', wave_data, RATE)
        else:
            print('non cough')

    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
