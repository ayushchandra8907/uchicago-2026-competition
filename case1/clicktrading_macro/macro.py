from threading import Lock

from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Controller as KeyboardController
from pynput import keyboard

import time


# Edit these screen positions yourself.
POS_B1 = ( 65, 636)
POS_S1 = (445, 636)
POS_B2 = (564, 636)
POS_S2 = (945, 636)
POS_B3 = (1054, 636)
POS_S3 = (1449, 636)


KEY_TO_POSITION = {
    "w": POS_B1,
    "e": POS_B2,
    "r": POS_B3,
    "i": POS_S1,
    "o": POS_S2,
    "p": POS_S3,
}

mouse_controller = MouseController()
keyboard_controller = KeyboardController()
state_lock = Lock()


def trigger_click(trigger_key: str) -> None:
    """Move to the configured position, click once, then send 40 + Enter."""
    position = KEY_TO_POSITION[trigger_key]
    mouse_controller.position = position
    time.sleep(.1)
    mouse_controller.click(Button.left, 1)
    time.sleep(.1) # sleep for .3 to wait for loading
    keyboard_controller.type("40")
    keyboard_controller.press(keyboard.Key.enter)
    keyboard_controller.release(keyboard.Key.enter)

    print(f"Key pressed: {trigger_key}")
    print(f"Clicked position: {position}")
    print('Typed "40" and pressed Enter')


def on_press(key):
    """Handle hotkeys for clicking and exiting."""
    if key == keyboard.Key.esc:
        print("ESC pressed. Exiting.")
        return False

    char = getattr(key, "char", None)
    if char is None:
        return None

    char = char.lower()
    if char not in KEY_TO_POSITION:
        return None

    # Lock so repeated presses do not overlap and break mouse movement/clicks.
    with state_lock:
        trigger_click(char)

    return None


if __name__ == "__main__":
    print("Click trading macro started.")
    print("Tracked keys: w, e, r, i, o, p")
    print("Press ESC to exit.")
    print("Note for macOS: you may need to enable Accessibility permissions")
    print("for your Terminal app or Python app in System Settings.")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
