import time
import ddsp
import ddsp.training
import gin
import numpy as np
import soundfile
from tensorflow.python.ops.numpy_ops import np_config


def timbre_transfer():
    print('Tranfering Timbre...')
    
    np_config.enable_numpy_behavior()
    
    ### READ AUDIO AND EXTRACT FEATURES ###
    audio, sr = soundfile.read('./out_loud.wav')
    if len(audio.shape) == 1:
        audio = audio[np.newaxis, :]

    # Setup the session.
    ddsp.spectral_ops.reset_crepe()

    # Compute features.
    start_time = time.time()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
    audio_features_mod = None
    print('Audio features took %.1f seconds' % (time.time() - start_time))

    ### LOAD MODEL ###
    print('Laoding model...')
    
    gin_file = './models/operative_config-0.gin'
    ckpt = './models/ckpt-40000'

    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)
        
    
    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size
    
    gin_params = [
        'Harmonic.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
        'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
    ]
    
    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config(gin_params)


    # Trim all input vectors to correct lengths 
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features['audio'] = audio_features['audio'][:, :n_samples]


    # Set up the model just to predict audio given new conditioning
    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)

    # Build model by running a batch through it.
    start_time = time.time()
    _ = model(audio_features, training=False)
    
    ### RESYNTHESIZE AUDIO ###
    print('Resynthesizing...')
    
    af = audio_features
    # Run a batch of predictions.
    outputs = model(af, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    
    ag = audio_gen.numpy()
    ag = np.reshape(ag, np.shape(ag[0]))
    soundfile.write('audio_gen.wav', ag, sr)
    print('Cough music generated!')