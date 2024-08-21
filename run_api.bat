@echo off

set GIT=
set PYTHON="C:\Program Files\Python310\python.exe"
set VENV_DIR=

::TIMEOUT /T 1

::call %PYTHON% -m pip install git+https://github.com/openai/CLIP.git
::call %PYTHON% -m spacy download en_core_web_sm
call %PYTHON% -m src.humorapid.main

pause
