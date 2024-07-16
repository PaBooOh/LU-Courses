#!/usr/bin/env python3

'''
This example generates the speech waveform directly from a gestural score.

Look into example1.py for more thorough comments on how to interface
vocaltractlab API from python3.

'''

import ctypes
import os
import shutil
import sys

# Use 'VocalTractLabApi32.dll' if you use a 32-bit python version.

if sys.platform == 'win32':
    VTL = ctypes.cdll.LoadLibrary('./VocalTractLabApi.dll')
else:
    VTL = ctypes.cdll.LoadLibrary('./VocalTractLabApi.so')


# get version / compile date
version = ctypes.c_char_p(b'                                ')
VTL.vtlGetVersion(version)
print('Compile date of the library: "%s"' % version.value.decode())

# Synthesize from a gestural score.
speaker_file_name = ctypes.c_char_p(b'JD2.speaker')
gesture_file_name = ctypes.c_char_p(b'ala.ges')
wav_file_name = ctypes.c_char_p(b'ala.wav')
numSamples = ctypes.c_int(0)
audio = (ctypes.c_double * int(1*44100))() #Reserve memory
# initialize vtl
failure = VTL.vtlInitialize(speaker_file_name)
if failure != 0:
    raise ValueError('Error in vtlInitialize! Errorcode: %i' % failure)

print('Calling gesToWav()...')

failure = VTL.vtlGesturalScoreToAudio(gesture_file_name,
                                      wav_file_name,
                                      ctypes.byref(audio),
                                      ctypes.byref(numSamples),
                                      int(1))

if failure != 0:
    raise ValueError('Error in vtlGesToWav! Errorcode: %i' % failure)

failure = VTL.vtlClose()

print('Finished.')




# fix wav header on non windows os
if sys.platform != 'win32':
    WAV_HEADER = (b'RIFF\x8c\x87\x00\x00WAVEfmt\x20\x10\x00\x00\x00\x01\x00\x01'
                  + b'\x00"V\x00\x00D\xac\x00\x00\x02\x00\x10\x00data')

    wav_file = wav_file_name.value.decode()
    with open(wav_file, 'rb') as file_:
        content = file_.read()

    shutil.move(wav_file, wav_file + '.bkup')

    with open(wav_file, 'wb') as newfile:
        newcontent = WAV_HEADER + content[68:]
        newfile.write(newcontent)

    os.remove(wav_file + '.bkup')

    print('Fixed header in %s.' % wav_file)

