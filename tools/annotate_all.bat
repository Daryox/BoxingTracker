@echo off
REM Annotate punches for each video, one at a time.
REM Run from the BoxingTracker root directory:
REM   tools\annotate_all.bat
REM
REM Controls in the annotation window:
REM   SPACE       play/pause
REM   LEFT/RIGHT  step 1 frame
REM   , / .       jump 5 frames
REM   TAB         switch active fighter (green=active, blue=inactive)
REM   S           mark punch START
REM   E           mark punch END then pick type (1-6)
REM               1=jab  2=cross  3=lead_hook  4=rear_hook
REM               5=lead_uppercut  6=rear_uppercut
REM   U           undo last clip
REM   Q           save and quit

set PYTHON=.venv\Scripts\python
set VIDEOS=logic\models\visual\punches\data\videos
set RAW=logic\models\visual\punches\data\raw
set TOOL=tools\annotate_punches.py

echo === Video 1/7: Amateur boxing sparring Round 1 ===
%PYTHON% %TOOL% "%VIDEOS%\Amateur boxing sparring Round 1.mp4" "%RAW%\V11_raw.npy" --id V11
echo Done. Press any key for next video...
pause > nul

echo === Video 2/7: BEGINNER BOXING SPARRING ===
%PYTHON% %TOOL% "%VIDEOS%\BEGINNER BOXING SPARRING!.mp4" "%RAW%\V12_raw.npy" --id V12
echo Done. Press any key for next video...
pause > nul

echo === Video 3/7: Imane Khalif vs Angela Carini ===
%PYTHON% %TOOL% "%VIDEOS%\Imane Khalif vs. Angela Carini Full Fight 2024 Olympic Boxing.mp4" "%RAW%\V13_raw.npy" --id V13
echo Done. Press any key for next video...
pause > nul

echo === Video 4/7: SHARP PRO BOXER ===
%PYTHON% %TOOL% "%VIDEOS%\SHARP! 9-0 PRO BOXER Spars With Dallas RISING STARS For 6 Rounds!.mp4" "%RAW%\V14_raw.npy" --id V14
echo Done. Press any key for next video...
pause > nul

echo === Video 5/7: videoplayback (1) ===
%PYTHON% %TOOL% "%VIDEOS%\videoplayback (1).mp4" "%RAW%\V15_raw.npy" --id V15
echo Done. Press any key for next video...
pause > nul

echo === Video 6/7: videoplayback (2) ===
%PYTHON% %TOOL% "%VIDEOS%\videoplayback (2).mp4" "%RAW%\V16_raw.npy" --id V16
echo Done. Press any key for next video...
pause > nul

echo === Video 7/7: videoplayback ===
%PYTHON% %TOOL% "%VIDEOS%\videoplayback.mp4" "%RAW%\V17_raw.npy" --id V17
echo Done.

echo.
echo === All annotations complete ===
echo Clips saved to: logic\models\visual\punches\data\skeleton\
echo Labels saved to: logic\models\visual\punches\data\labels\
echo.
echo Now retrain with:
echo   .venv\Scripts\python -m logic.models.visual.punches.TCN_train
pause
