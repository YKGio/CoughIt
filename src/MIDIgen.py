import magenta
import note_seq
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
from pretty_midi import pretty_midi

def midi_generate():
      print('Generatin MIDI file...')
      ### MODEL ###
      music_vae = TrainedModel(
            configs.CONFIG_MAP['cat-mel_2bar_big'], 
            batch_size=4, 
            checkpoint_dir_or_path='./models/mel_2bar_big.ckpt')


      ### GENERATE ###
      generated_sequences = music_vae.sample(n=1, length=80, temperature=1.0)
      ns1 = generated_sequences[0]
      note_seq.sequence_proto_to_midi_file(ns1, './MIDI/generated.mid')

      print('MIDI file generated!')