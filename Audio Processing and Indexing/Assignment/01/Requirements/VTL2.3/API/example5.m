%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This example simply shows how to 
% o transform a segment sequence file into a gestural score file,
% o transform the gestural score file into a tract parameter sequence file,
% o generate audio from the tract parameter sequence file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% File name of the dll and header file (they differ only in the extension).

libName = 'VocalTractLabApi';

if ~libisloaded(libName)
    % To load the library, specify the name of the DLL and the name of the
    % header file. If no file extensions are provided (as below)
    % LOADLIBRARY assumes that the DLL ends with .dll and the header file
    % ends with .h.
    loadlibrary(libName, libName);
    disp(['Loaded library: ' libName]);
    pause(1);
end

if ~libisloaded(libName)
    error(['Failed to load external library: ' libName]);
    success = 0;
    return;
end

% *****************************************************************************
% list the methods
% *****************************************************************************

libfunctions(libName);   

% *****************************************************************************
% Print the version (compile date) of the library.
%
% void vtlGetVersion(char *version);
% *****************************************************************************

% Init the variable version with enough characters for the version string
% to fit in.
version = '                                ';
version = calllib(libName, 'vtlGetVersion', version);

disp(['Compile date of the library: ' version]);

% *****************************************************************************
% Initialize the VTL synthesis with the given speaker file name.
%
% void vtlInitialize(const char *speakerFileName)
% *****************************************************************************

speakerFileName = 'JD2.speaker';

failure = calllib(libName, 'vtlInitialize', speakerFileName);
if (failure ~= 0)
    disp('Error in vtlInitialize()!');   
    return;
end

% *****************************************************************************
% Get some constants.
%
% void vtlGetConstants(int *audioSamplingRate, int *numTubeSections,
%   int *numVocalTractParams, int *numGlottisParams);
% *****************************************************************************

audioSamplingRate = 0;
numTubeSections = 0;
numVocalTractParams = 0;
numGlottisParams = 0;

[failure, audioSamplingRate, numTubeSections, numVocalTractParams, numGlottisParams] = ...
    calllib(libName, 'vtlGetConstants', audioSamplingRate, numTubeSections, numVocalTractParams, numGlottisParams);

disp(['Audio sampling rate = ' num2str(audioSamplingRate)]);
disp(['Num. of tube sections = ' num2str(numTubeSections)]);
disp(['Num. of vocal tract parameters = ' num2str(numVocalTractParams)]);
disp(['Num. of glottis parameters = ' num2str(numGlottisParams)]);

% *****************************************************************************
% int vtlSegmentSequenceToGesturalScore(const char *segFileName, 
%   const char *gesFileName);
% *****************************************************************************

segFileName = 'apfelsine.seg';
gesFileName = 'apfelsine.ges';
tractSequenceFileName = 'apfelsine.txt';
wavFileName = 'apfelsine.wav';

failed = ...
  calllib(libName, 'vtlSegmentSequenceToGesturalScore', segFileName, gesFileName);

audio = zeros(10*44100, 1);     % Reserve memory for 10 s
numSamples = 0;

% *****************************************************************************
% int vtlGesturalScoreToAudio(const char* gesFileName, const char* wavFileName,
%  double* audio, int* numSamples, int enableConsoleOutput);
% *****************************************************************************

% Possibly synthesize the audio directly from the gestural score...

%[failed, gesFileName, wavFileName, audio, numSamples] = ...
%  calllib(libName, 'vtlGesturalScoreToAudio', gesFileName, wavFileName, audio, numSamples, 1);
%plot(1:numSamples, audio(1:numSamples));

% *****************************************************************************
% int vtlGesturalScoreToTractSequence(const char* gesFileName, 
%  const char* tractSequenceFileName);
% *****************************************************************************

failed = ...
  calllib(libName, 'vtlGesturalScoreToTractSequence', gesFileName, tractSequenceFileName);

% *****************************************************************************
% *****************************************************************************

% Here, the sequence of parameter values of the vocal tract or glottis models 
% could be manipulated in the tract sequence TXT file.

% *****************************************************************************
% int vtlTractSequenceToAudio(const char* tractSequenceFileName,
%  const char* wavFileName, double* audio, int* numSamples);
% *****************************************************************************

[failed, tractSequenceFileName, wavFileName, audio, numSamples] = ...
  calllib(libName, 'vtlTractSequenceToAudio', tractSequenceFileName, wavFileName, audio, numSamples);

plot(1:numSamples, audio(1:numSamples));


% *****************************************************************************
% Close the VTL synthesis.
%
% void vtlClose();
% *****************************************************************************

calllib(libName, 'vtlClose');

unloadlibrary(libName);

