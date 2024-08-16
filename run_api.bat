@echo off

set GIT=
set PYTHON="C:\Program Files\Python310\python.exe"
set VENV_DIR=C:\Users\Mr. RIAH\Documents\GenAI\sd_env

::TIMEOUT /T 1

call %PYTHON% -m pip install git+https://github.com/openai/CLIP.git
call %PYTHON% -m spacy download en_core_web_sm
call %PYTHON% -m api.main

pause
