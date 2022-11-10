from src import MIDIgen, MIDItoWAV, TimTrans
import soundfile
import warnings
warnings.filterwarnings("ignore")

MIDIgen.midi_generate()
MIDItoWAV.midi2wav()
TimTrans.timbre_transfer()