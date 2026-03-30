@echo off
REM Extract skeletons from all videos in data\videos\.
REM Run from the BoxingTracker root directory:
REM   tools\extract_all.bat

set PYTHON=.venv\Scripts\python
set VIDEOS=logic\models\visual\punches\data\videos
set TOOL=tools\extract_skeletons.py

echo === Extracting skeletons from all videos ===
echo.

%PYTHON% %TOOL% "%VIDEOS%\Amateur boxing sparring Round 1.mp4" --id V11
%PYTHON% %TOOL% "%VIDEOS%\BEGINNER BOXING SPARRING!.mp4"                                --id V12 
%PYTHON% %TOOL% "%VIDEOS%\Imane Khalif vs. Angela Carini Full Fight 2024 Olympic Boxing.mp4" --id V13 
%PYTHON% %TOOL% "%VIDEOS%\SHARP! 9-0 PRO BOXER Spars With Dallas RISING STARS For 6 Rounds!.mp4" --id V14 
%PYTHON% %TOOL% "%VIDEOS%\videoplayback (1).mp4"                                         --id V15 
%PYTHON% %TOOL% "%VIDEOS%\videoplayback (2).mp4"                                         --id V16 
%PYTHON% %TOOL% "%VIDEOS%\videoplayback.mp4"                                              --id V17 

echo.
echo === All extractions done ===
echo Raw skeleton files saved to: logic\models\visual\punches\data\raw\
pause
