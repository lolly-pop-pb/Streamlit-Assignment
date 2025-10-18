import os
from subprocess import Popen, PIPE, STDOUT
import sys
import time
import webbrowser

def main():
    # Getting path to python executable (full path of deployed python on Windows)
    executable = sys.executable
    path_to_main = os.path.join(os.path.dirname(__file__), "home.py")

    # Running streamlit server in a subprocess
    proc = Popen(
        [
            executable,
            "-m", "streamlit", "run", path_to_main,
            "--server.headless=true",
            "--global.developmentMode=false",
            "--server.port=8501",
            "--server.address=127.0.0.1",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
    )
    proc.stdin.close()

    # Wait a few seconds for the Streamlit server to start
    time.sleep(3)
    webbrowser.open("http://127.0.0.1:8501")

    # Print Streamlit logs in real-time
    while True:
        s = proc.stdout.readline()
        if not s:
            break
        print(s, end="")

    proc.wait()

if __name__ == "__main__":
    main()