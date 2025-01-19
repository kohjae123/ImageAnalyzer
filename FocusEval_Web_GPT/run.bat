@echo off
echo FocusEval 실행을 시작합니다...

:: venv가 있는지 확인
if not exist "venv" (
    echo 가상환경을 생성합니다...
    python -m venv venv
    call venv\Scripts\activate
    echo 필요한 패키지를 설치합니다...
    pip install -r backend\requirements.txt
) else (
    call venv\Scripts\activate
)

:: 브라우저 실행 (약간 지연을 주어 서버가 먼저 시작되게 함)
start "" cmd /c "timeout /t 3 && start http://localhost:8000"

:: FastAPI 서버 실행
echo FastAPI 서버를 시작합니다...
uvicorn backend.main:app --reload

pause 