#Test synthBlock function with Python, modified by Anqi Xu (a.xu.17@ucl.ac.uk) 10/03/2020

import os, sys, ctypes #neccessary packages, no need to install
import numpy as np, matplotlib.pyplot as plt 
from scipy.io import wavfile
#Initilization
#Load VocalTractLab library
VTL = ctypes.cdll.LoadLibrary('./VocalTractLabApi.dll')
#Get version and compile date
version = ctypes.c_char_p(b'                                ')
VTL.vtlGetVersion(version)
print('Compile date of the library: "%s"' % version.value.decode())
#initialize VTL
speaker_file_name = ctypes.c_char_p('JD2.speaker'.encode())
failure = VTL.vtlInitialize(speaker_file_name) 
if failure != 0:
    raise ValueError('Error in getting speaker file! Errorcode: %i' % failure)
#Get some constants
audio_sampling_rate = ctypes.c_int(0)
number_tube_sections = ctypes.c_int(0)
number_vocal_tract_parameters = ctypes.c_int(0)
number_glottis_parameters = ctypes.c_int(0)

VTL.vtlGetConstants(ctypes.byref(audio_sampling_rate),
                    ctypes.byref(number_tube_sections),
                    ctypes.byref(number_vocal_tract_parameters),
                    ctypes.byref(number_glottis_parameters))

print('Audio sampling rate = %i' % audio_sampling_rate.value)
print('Num. of tube sections = %i' % number_tube_sections.value) 
print('Num. of vocal tract parameters = %i' % number_vocal_tract_parameters.value) 
print('Num. of glottis parameters = %i' % number_glottis_parameters.value) 

# Get the parameters of the vocal tract model
# Hint: Reserve 32 chars for each parameter.
TRACT_PARAM_TYPE = ctypes.c_double * number_vocal_tract_parameters.value
tract_param_names = ctypes.c_char_p((' ' * 32 * number_vocal_tract_parameters.value).encode())
tract_param_min = TRACT_PARAM_TYPE()
tract_param_max = TRACT_PARAM_TYPE()
tract_param_neutral = TRACT_PARAM_TYPE()

VTL.vtlGetTractParamInfo(tract_param_names,
                         ctypes.byref(tract_param_min),
                         ctypes.byref(tract_param_max),
                         ctypes.byref(tract_param_neutral))
						 
print('Vocal tract parameters: "%s"' % tract_param_names.value.decode()) 
print('Vocal tract parameter minima: ' + str(list(tract_param_min))) 
print('Vocal tract parameter maxima: ' + str(list(tract_param_max))) 
print('Vocal tract parameter neutral: ' + str(list(tract_param_neutral))) 

# Get information about the parameters of glottis model
# Hint: Reserve 32 chars for each parameter.
GLOTTIS_PARAM_TYPE = ctypes.c_double * number_glottis_parameters.value
glottis_param_names = ctypes.c_char_p((' ' * 32 * number_glottis_parameters.value).encode())
glottis_param_min = GLOTTIS_PARAM_TYPE()
glottis_param_max = GLOTTIS_PARAM_TYPE()
glottis_param_neutral = GLOTTIS_PARAM_TYPE()

VTL.vtlGetGlottisParamInfo(glottis_param_names,
                           ctypes.byref(glottis_param_min),
                           ctypes.byref(glottis_param_max),
                           ctypes.byref(glottis_param_neutral))

print('Glottis parameters: "%s"' % glottis_param_names.value.decode()) 
print('Glottis parameter minima: ' + str(list(glottis_param_min))) 
print('Glottis parameter maxima: ' + str(list(glottis_param_max))) 
print('Glottis parameter neutral: ' + str(list(glottis_param_neutral))) 

# Get the vocal tract parameter values for the vocal tract shapes of /i/ and /a/, which are saved in the speaker file.
shape_name = ctypes.c_char_p(b'a')
params_a = TRACT_PARAM_TYPE()
failure = VTL.vtlGetTractParams(shape_name, ctypes.byref(params_a))

if failure != 0:
    raise ValueError('Error in getting vowel shape a! Errorcode: %i' % failure)
	
shape_name = ctypes.c_char_p(b'i')
params_i = TRACT_PARAM_TYPE()
failure = VTL.vtlGetTractParams(shape_name, ctypes.byref(params_i))
if failure != 0:
    raise ValueError('Error in getting vowel shape i! Errorcode: %i' % failure)
	
del shape_name

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# synthesize transition from i to a
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
duration_s = 1  # seconds
frame_rate = int(220)  # Hz
number_frames = int(duration_s * frame_rate)

audio = (ctypes.c_double * int((number_frames-1) * frame_rate))()# new!!!
enableConsoleOutput = int(1)

#create empty arrays
tract_params = (ctypes.c_double * (number_frames * number_vocal_tract_parameters.value))()
glottis_params = (ctypes.c_double * (number_frames * number_glottis_parameters.value))()

# Create the vocal tract shapes that slowly change from /a/ to /i/ from the first to the last frame.
for ii in range(number_frames):
    # transition linearly from /a/ to /i/
    dd = ii / (number_frames - 1)
    for jj in range(number_vocal_tract_parameters.value):
        tract_params[ii * number_vocal_tract_parameters.value + jj] =\
                (1 - dd) * params_a[jj] + dd * params_i[jj]

    offset_glottis = ii * number_glottis_parameters.value
    # transition F0 in Hz going from 120 Hz down to 100 Hz
    glottis_params[offset_glottis + 0] = 120.0 - 20.0 * (ii / number_frames)

    # Start with zero subglottal pressure and then go to 1000 Pa.
    if ii <= 1:
        glottis_params[offset_glottis + 1] = 0.0
    elif ii == 2:
        glottis_params[offset_glottis + 1] = 4000.0
    else:
        glottis_params[offset_glottis + 1] = 8000.0

    # use the neutral settings for the rest of glottis parameters
    for jj in range(2, number_glottis_parameters.value):
        glottis_params[offset_glottis + jj] = glottis_param_neutral[jj]


# Call the synthesis function. 
failure = VTL.vtlSynthBlock(ctypes.byref(tract_params),  # input
                            ctypes.byref(glottis_params),  # input
                            number_frames,  # input
                            frame_rate,  # input
                            ctypes.byref(audio),  # output
                            enableConsoleOutput)  # output

if failure != 0:
    raise ValueError('Error in vtlSynthBlock! Errorcode: %i' % failure)

print('First 40 data points for every vector:')
print('  vocal_tract_pram: %s' % str(list(tract_params)[:40]))
print('  glottis_tract_pram: %s' % str(list(glottis_params)[:40]))
print('  number_of_frames: %s' %  number_frames)
print('  frame_rate: %s' %  frame_rate)
print('  audio: %s' % str(list(audio)[:40]))

#Exit
VTL.vtlClose()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plot and save the audio signal
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if np is not None:
    wav = np.array(audio)

if wavfile is not None and np is not None:
    wav_int = np.int16(wav  * (2 ** 15 - 1))
    wavfile.write('ai_test.wav', audio_sampling_rate.value, wav_int)
    print('saved audio to "ai_test.wav"')
else:
    print('scipy not available')
    print('skip writing out wav file')

