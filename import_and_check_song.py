from pydub import AudioSegment
import os 
from matplotlib import pyplot as plt
import IPython.display as ipd
import librosa 

file_name = input("Please upload a song: ") 
 
try: 
    # Attempt to open the file in read mode 
    with open(file_name, 'r') as file:   
        song = file.read()   
except FileNotFoundError: 
    print(f"The file '{file_name}' does not exist.") 
except Exception as e: 
    print(f"An error occurred while trying to read the file: {e}") 

# Create an AudioSegment instance
wav_file = AudioSegment.from_file(song, format="wav")
mp3_file = AudioSegment.from_file(song, format="mp3")

# Check the type
print(type(wav_file))
print(type(mp3_file))

# source code: https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonAudio.html
def print_plot_play(x, Fs, text):
    """1. Prints information about an audio singal, 2. plots the waveform, and 3. Creates player
    
    Args: 
        x: Input signal
        Fs: Sampling rate of x    
        text: Text to print
    """
    print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, Fs, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=Fs))

# Read wav
fn_wav = os.path.join('..', 'data', 'B', wav_file)
x, Fs = librosa.load(fn_wav, sr=None)
print_plot_play(x=x, Fs=Fs, text='WAV file: ')

# Read mp3
fn_mp3 = os.path.join('..', 'data', 'B', mp3_file)
x, Fs = librosa.load(fn_mp3, sr=None)
print_plot_play(x=x, Fs=Fs, text='MP3 file: ')