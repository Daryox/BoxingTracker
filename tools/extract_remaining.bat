@echo off
REM Extracts skeletons for remaining sparring videos.
REM Run from the BoxingTracker root directory.

set PYTHON=.venv\Scripts\python
set VIDEOS=logic\models\visual\punches\data\videos
set TOOL=tools\extract_skeletons.py

%PYTHON% %TOOL% "%VIDEOS%\BEGINNER BOXING SPARRING!.mp4"                                --id V12 
%PYTHON% %TOOL% "%VIDEOS%\Imane Khalif vs. Angela Carini Full Fight 2024 Olympic Boxing.mp4" --id V13 
%PYTHON% %TOOL% "%VIDEOS%\SHARP! 9-0 PRO BOXER Spars With Dallas RISING STARS For 6 Rounds!.mp4" --id V14 
%PYTHON% %TOOL% "%VIDEOS%\videoplayback (1).mp4"                                         --id V15 
%PYTHON% %TOOL% "%VIDEOS%\videoplayback (2).mp4"                                         --id V16 
%PYTHON% %TOOL% "%VIDEOS%\videoplayback.mp4"                                              --id V17 

echo.
echo === Done ===
pause
