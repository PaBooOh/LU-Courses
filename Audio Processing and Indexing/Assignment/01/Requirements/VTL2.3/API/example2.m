%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This example generates the speech waveform directly from a gestural
% score.
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
% Synthesize from a gestural score.
%
% int vtlGesturalScoreToWav(const char *gesFileName, const char *wavFileName,
%  int enableConsoleOutput);
% *****************************************************************************

speakerFileName = 'JD2.speaker';
gestureFileName = 'ala.ges';
wavFileName = 'ala.wav';

failure = calllib(libName, 'vtlInitialize', speakerFileName);

if (failure ~= 0)
    disp('Error in vtlInitialize()! Error code:');
    failure
    return;
end

numSamples = 0;
audio = zeros(44100, 0);   % Enough for 1 s of audio.

failure = calllib(libName, 'vtlGesturalScoreToAudio', gestureFileName, ...
    wavFileName, audio, numSamples, 1);

if (failure ~= 0)
    disp('Error in vtlGesturalScoreToAudio()! Error code:');
    failure
    return;
end

failure = calllib(libName, 'vtlClose');

disp('Finished.');

% Play the synthesized wav file.
s = audioread(wavFileName);

plot(1:length(s), s);

sound(s, 44100);


