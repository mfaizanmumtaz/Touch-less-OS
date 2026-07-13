"""Maps recognized gestures to OS-level app-launch actions (Windows)."""

import subprocess

GESTURE_TO_APP = {
    "open_palm": ("Chrome", 'start "" chrome'),
    "fist": ("VS Code", "code"),
    "peace": ("File Explorer", "explorer"),
    "thumbs_up": ("Calculator", "calc"),
    "square_frame": ("ChatGPT", 'start "" chrome "https://chat.openai.com"'),
}


def run_action(gesture):
    if gesture not in GESTURE_TO_APP:
        return None
    app_name, command = GESTURE_TO_APP[gesture]
    try:
        subprocess.Popen(command, shell=True)
    except OSError as e:
        print(f"Failed to launch {app_name}: {e}")
        return None
    return app_name
