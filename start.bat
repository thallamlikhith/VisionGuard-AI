@echo off
REM The one launcher for this project: it starts both servers.
REM   backend  - FastAPI on http://127.0.0.1:8090
REM   frontend - Vite dev server on http://127.0.0.1:5174
REM Close the two console windows to stop both.
REM
REM Everything is checked before anything is started, and if the frontend does
REM not come up the backend this script started is stopped again, so a failed
REM start never leaves half a demo running.
REM When both servers answer, the dashboard is opened in Edge or Chrome by full
REM path - never through the Windows URL handler, which on some machines is a
REM broken install and answers with a side-by-side error dialog. Set NO_BROWSER=1
REM to skip opening anything.
REM
REM Re-runnable: if a previous backend/frontend is already answering on its
REM port, it is reused instead of failing with "port already in use", and a
REM failed start only stops the servers this run actually started.
REM Usage: start.bat [--no-browser] [--reinstall-frontend]

setlocal EnableExtensions
cd /d "%~dp0"

set "BACKEND_PORT=8090"
set "FRONTEND_PORT=5174"
set "FRONTEND_URL=http://127.0.0.1:%FRONTEND_PORT%/"
set "BACKEND_PID="
set "FRONTEND_PID="
set "BACKEND_REUSED="
set "FRONTEND_REUSED="
set "PIDFILE=%TEMP%\trinetra_pids.txt"

REM Optional flags, so double-click stays the same but scripts can pass them.
for %%A in (%*) do (
    if /i "%%A"=="--no-browser" set "NO_BROWSER=1"
    if /i "%%A"=="--reinstall-frontend" set "REINSTALL_FRONTEND=1"
)

echo.
echo   Trinetra AI
echo   ----------------------
echo   Starting both servers.
echo.
echo   Backend  : http://127.0.0.1:%BACKEND_PORT%
echo   Frontend : %FRONTEND_URL%
echo.

REM Prefer the project venv when it exists, so a system Python without the
REM deps never shadows the working interpreter.
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

where node >nul 2>&1
if errorlevel 1 (
    echo ERROR: node was not found on PATH. Install Node.js 18 or newer and retry.
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm was not found on PATH. Reinstall Node.js so npm comes with it.
    pause
    exit /b 1
)

"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: could not run "%PYTHON% --version". Install Python 3.10+ and retry.
    pause
    exit /b 1
)

REM Backend deps may be missing after a fresh clone or pull. Fail here with the
REM fix instead of opening a backend window that exits immediately.
"%PYTHON%" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Backend packages missing - installing requirements.txt with %PYTHON%...
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: pip install failed. Read the messages above, then retry.
        pause
        exit /b 1
    )
    "%PYTHON%" -c "import fastapi, uvicorn" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: fastapi/uvicorn still missing after install. Activate the right venv and retry.
        pause
        exit /b 1
    )
)

if defined REINSTALL_FRONTEND (
    echo Reinstalling frontend dependencies...
    if exist "frontend\node_modules" rmdir /s /q "frontend\node_modules"
)

if not exist "frontend\node_modules" (
    echo Installing frontend dependencies ^(first run only^)...
    call npm --prefix frontend install
    if errorlevel 1 (
        echo ERROR: npm install failed. Read the messages above, then retry.
        pause
        exit /b 1
    )
)

REM A port can be held by a previous run of this same script. The HTTP probe
REM is authoritative: if our own healthy server answers, reuse it. Only a
REM listener that answers nothing is an error. Probe first because netstat
REM formatting can miss IPv6 listeners.
powershell -NoProfile -Command "try { $null = Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:%BACKEND_PORT%/health -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo Reusing the backend already running on port %BACKEND_PORT%.
    set "BACKEND_REUSED=1"
    goto :backend_port_done
)
netstat -ano | findstr /R /C:":%BACKEND_PORT% .*LISTENING" >nul 2>&1
if errorlevel 1 goto :backend_port_done
echo ERROR: port %BACKEND_PORT% is already in use by another program. Close it and retry.
pause
exit /b 1
:backend_port_done

powershell -NoProfile -Command "try { $null = Invoke-WebRequest -UseBasicParsing -Uri %FRONTEND_URL% -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo Reusing the frontend already running on port %FRONTEND_PORT%.
    set "FRONTEND_REUSED=1"
    goto :frontend_port_done
)
netstat -ano | findstr /R /C:":%FRONTEND_PORT% .*LISTENING" >nul 2>&1
if errorlevel 1 goto :frontend_port_done
echo ERROR: port %FRONTEND_PORT% is already in use by another program. Close it and retry.
pause
exit /b 1
:frontend_port_done

REM The Windows "side-by-side configuration is incorrect" dialog comes from a
REM damaged native file in node_modules (esbuild.exe or Rollup's addon). Both are
REM run for real here, before anything is started, so the cause is named instead
REM of leaving a dead port behind.
call npm --prefix frontend run check
if errorlevel 1 (
    echo.
    echo ERROR: the frontend cannot start on this machine - see the fix above.
    pause
    exit /b 1
)

if not exist "yolo11n.pt" (
    echo WARNING: yolo11n.pt is missing - the detector will download it on first run.
)

if defined BACKEND_REUSED goto :backend_ready
call :launch "BACKEND" cmd.exe "/k %PYTHON% -m uvicorn api.server:app --host 127.0.0.1 --port %BACKEND_PORT%"

echo Waiting for the backend to come up...
for /l %%I in (1,1,40) do (
    powershell -NoProfile -Command "try { $null = Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:%BACKEND_PORT%/health -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 goto :backend_ready
    REM `timeout` refuses to run when stdin is redirected, which is exactly what
    REM happens in a scripted check, so the wait is done with ping instead.
    ping -n 2 127.0.0.1 >nul
)
echo ERROR: the backend did not start. Read the "Trinetra AI API" window.
call :stop "BACKEND"
pause
exit /b 1

:backend_ready
echo Backend is up.

if defined FRONTEND_REUSED goto :frontend_ready
call :launch "FRONTEND" cmd.exe "/k call npm --prefix frontend run dev"

echo Waiting for the dashboard on port %FRONTEND_PORT%...
for /l %%I in (1,1,60) do (
    powershell -NoProfile -Command "try { $null = Invoke-WebRequest -UseBasicParsing -Uri %FRONTEND_URL% -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 goto :frontend_ready
    REM `timeout` refuses to run when stdin is redirected, which is exactly what
    REM happens in a scripted check, so the wait is done with ping instead.
    ping -n 2 127.0.0.1 >nul
)
echo.
echo ERROR: Vite is not answering on port %FRONTEND_PORT%.
echo   Most likely cause: a damaged esbuild.exe or Rollup addon in frontend\node_modules.
echo   Fix: close this window, delete frontend\node_modules, then run start.bat again.
echo   The backend started here has been stopped again.
call :stop "FRONTEND"
REM Only undo what this run started: a reused backend belongs to someone else.
if not defined BACKEND_REUSED call :stop "BACKEND"
pause
exit /b 1

:frontend_ready
REM NO_BROWSER=1 is for scripted checks, which must not open a window.
if defined NO_BROWSER goto :no_browser
if defined TRINETRA_NO_BROWSER goto :no_browser
call :open_browser
goto :no_browser

:no_browser
echo.
echo Both servers are up.
echo   Frontend %FRONTEND_URL%
echo   Backend  http://127.0.0.1:%BACKEND_PORT%
echo Close both console windows to stop.
echo.
pause
exit /b 0


REM ---------------------------------------------------------------------------
REM :open_browser
REM   Opens the dashboard in a browser named by its full path, never by handing
REM   the URL to the shell. `start "" <url>` makes Windows look up the
REM   registered http handler and run whatever it points at: on this machine that
REM   is a per-user Chrome whose manifest names a dependent assembly that cannot
REM   be resolved, so Windows answers with the "side-by-side configuration is
REM   incorrect" dialog and no browser at all. Naming the executable skips that
REM   lookup entirely.
REM   Edge first, then Chrome, then the per-user copies, because a per-user copy
REM   is the one that can be half installed. Nothing here changes a setting: the
REM   handler is left exactly as it is. If no browser exists the URL is printed
REM   for the user to open by hand and the servers stay up either way.
REM ---------------------------------------------------------------------------

:open_browser
set "BROWSER="
if not defined BROWSER if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "BROWSER=%LocalAppData%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%LocalAppData%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%LocalAppData%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER goto :no_browser_installed
start "" "%BROWSER%" "%FRONTEND_URL%"
exit /b 0

:no_browser_installed
echo.
echo No Edge or Chrome was found, so nothing was opened.
echo Open this link in your browser: %FRONTEND_URL%
exit /b 0


REM ---------------------------------------------------------------------------
REM :launch <name> <program> <args>
REM   Starts the program in its own console window and records its PID, so a
REM   failed start can be undone with :stop. The PID goes through a file because
REM   reading it back out of the PowerShell output would mean parsing quotes.
REM :stop <name>
REM   Kills the window and everything started inside it. Never kills a reused
REM   server: those PIDs stay empty, so :stop is a no-op for them.
REM ---------------------------------------------------------------------------

:launch
set "PID_NAME=%~1_PID"
set "PID_LINE="
powershell -NoProfile -Command "$p = Start-Process -FilePath '%~2' -ArgumentList '%~3' -WorkingDirectory '%CD%' -PassThru; Set-Content -Path '%PIDFILE%' -Value ('%1=' + $p.Id)"
set /p PID_LINE=<"%PIDFILE%"
for /f "tokens=1,* delims==" %%A in ("%PID_LINE%") do set "%PID_NAME%=%%B"
del /q "%PIDFILE%" >nul 2>&1
if not defined PID_LINE exit /b 1
exit /b 0

:stop
if "%~1"=="BACKEND" (
    if defined BACKEND_PID taskkill /PID %BACKEND_PID% /T /F >nul 2>&1
    set "BACKEND_PID="
)
if "%~1"=="FRONTEND" (
    if defined FRONTEND_PID taskkill /PID %FRONTEND_PID% /T /F >nul 2>&1
    set "FRONTEND_PID="
)
exit /b 0
