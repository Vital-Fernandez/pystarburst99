import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).parent / "app" / "streamlit_app.py"


def main():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.exit("The interface requires the optional dependencies:\n"
                 "    pip install 'pystarburst99[full]'")

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    sys.exit(subprocess.call(cmd))