%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This example generates the transition from /a/ to /i/ using the 
% incremental tract synthesis with the function 
% vtlSynthesisAddTract(...).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% File name of the dll and header file (they differ only in the extension)

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
% Get information about the parameters of the vocal tract model and the
% glottis model.
%
% void vtlGetTractParamInfo(char *names, double *paramMin, double *paramMax, 
%   double *paramNeutral);
% void vtlGetGlottisParamInfo(char *names, double *paramMin, double *paramMax, 
%   double *paramNeutral);
% *****************************************************************************

% Reserve 32 chars for each parameter.
tractParamNames = blanks(numVocalTractParams*32);
tractParamMin = zeros(1, numVocalTractParams);
tractParamMax = zeros(1, numVocalTractParams);
tractParamNeutral = zeros(1, numVocalTractParams);

[failure, tractParamNames, tractParamMin, tractParamMax, tractParamNeutral] = ...
  calllib(libName, 'vtlGetTractParamInfo', tractParamNames, tractParamMin, ...
  tractParamMax, tractParamNeutral);
    
% Reserve 32 chars for each parameter.
glottisParamNames = blanks(numGlottisParams*32);
glottisParamMin = zeros(1, numGlottisParams);
glottisParamMax = zeros(1, numGlottisParams);
glottisParamNeutral = zeros(1, numGlottisParams);

[failure, glottisParamNames, glottisParamMin, glottisParamMax, glottisParamNeutral] = ...
  calllib(libName, 'vtlGetGlottisParamInfo', glottisParamNames, glottisParamMin, ...
  glottisParamMax, glottisParamNeutral);

disp(['Vocal tract parameters: ' tractParamNames]);
disp(['Glottis parameters: ' glottisParamNames]);

% *****************************************************************************
% Get the vocal tract parameter values for the vocal tract shapes of /i/
% and /a/, which are saved in the speaker file.
%
% int vtlGetTractParams(char *shapeName, double *param);
% *****************************************************************************

shapeName = 'a';
paramsA = zeros(1, numVocalTractParams);
[failed, shapeName, paramsA] = ...
  calllib(libName, 'vtlGetTractParams', shapeName, paramsA);

if (failed ~= 0)
    disp('Error: Vocal tract shape "a" not in the speaker file!');   
    return;
end

shapeName = 'i';
paramsI = zeros(1, numVocalTractParams);
[failed, shapeName, paramsI] = ...
  calllib(libName, 'vtlGetTractParams', shapeName, paramsI);

if (failed ~= 0)
    disp('Error: Vocal tract shape "i" not in the speaker file!');   
    return;
end

% *****************************************************************************
% Incrementally synthesize a transition from /a/ to /i/.
%
% void vtlSynthesisReset()
%
% int vtlSynthesisAddTract(int numNewSamples, double *audio,
%   double *tractParams, double *glottisParams);
% *****************************************************************************

audio1 = zeros(1, 10000);
audio2 = zeros(1, 10000);
audio3 = zeros(1, 10000);
audio4 = zeros(1, 10000);

glottisParams = glottisParamNeutral;

% Initialize the tube synthesis.
calllib(libName, 'vtlSynthesisReset');

% Submit the initial vocal tract shape (numSamples=0) with P_sub = 0
glottisParams(2) = 0;       % P_sub = 0 dPa
[failure, audio1, tractParams, glottisParams] = ...
  calllib(libName, 'vtlSynthesisAddTract', 0, audio1, ...
    paramsI, glottisParams);

% Ramp up the subglottal pressure within 1000 samples
glottisParams(2) = 8000;   % P_sub = 8000 dPa
[failure, audio1, tractParams, glottisParams] = ...
  calllib(libName, 'vtlSynthesisAddTract', 1000, audio1, ...
    paramsI, glottisParams);

% Make transitions between /a/ and /i/
[failure, audio2, tractParams, glottisParams] = ...
  calllib(libName, 'vtlSynthesisAddTract', 10000, audio2, ...
    paramsA, glottisParams);

[failure, audio3, tractParams, glottisParams] = ...
  calllib(libName, 'vtlSynthesisAddTract', 10000, audio3, ...
    paramsI, glottisParams);

[audio4, tractParams, glottisParams] = ...
  calllib(libName, 'vtlSynthesisAddTract', 10000, audio4, ...
    paramsA, glottisParams);

audio = [audio1(1:1000) audio2 audio3 audio4];

% Plot and play the audio signal

plot(audio);
soundsc(audio, double(audioSamplingRate));
audiowrite('test.wav', audio, audioSamplingRate);

% *****************************************************************************
% Close the VTL synthesis.
%
% void vtlClose();
% *****************************************************************************

calllib(libName, 'vtlClose');

unloadlibrary(libName);
