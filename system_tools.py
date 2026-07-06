import shutil
import subprocess


def _run(cmd):
    "run command return ok"
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        )
        return out.returncode == 0, out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False, ""

_screen_asleep = False


def _sleep_screen():
    # Wayland/GNOME: use loginctl to lock the session (blanks + locks)
    if shutil.which("loginctl"):
        ok, _ = _run(["loginctl", "lock-session"])
        if ok:
            return True
    # Fallback: xdg-screensaver
    if shutil.which("xdg-screensaver"):
        ok, _ = _run(["xdg-screensaver", "activate"])
        if ok:
            return True
    # Last resort: X11 DPMS (only works on pure X11 with DPMS extension)
    if shutil.which("xset"):
        _run(["xset", "dpms", "force", "off"])
        return True
    return False


def _wake_screen():
    """Wake / unlock the screen using whatever is available."""
    if shutil.which("loginctl"):
        ok, _ = _run(["loginctl", "unlock-session"])
        if ok:
            return True
    if shutil.which("xdg-screensaver"):
        ok, _ = _run(["xdg-screensaver", "reset"])
        if ok:
            return True
    if shutil.which("xset"):
        _run(["xset", "dpms", "force", "on"])
        _run(["xset", "s", "reset"])
        return True
    return False


def toggle_screen():
    """Sleep the screen if awake, wake it if asleep. Returns the new state str."""
    global _screen_asleep

    if _screen_asleep:
        ok = _wake_screen()
        if not ok:
            print("[system_tools] no tool found to wake the screen")
            return "unknown"
        _screen_asleep = False
        return "awake"
    else:
        ok = _sleep_screen()
        if not ok:
            print("[system_tools] no tool found to sleep the screen")
            return "unknown"
        _screen_asleep = True
        return "asleep"

_DEFAULT_SINK = "@DEFAULT_SINK@"


def _is_muted():
    """Return True/False if known, else None."""
    ok, out = _run(["pactl", "get-sink-mute", _DEFAULT_SINK])
    if not ok:
        return None
    return "yes" in out.lower()


def toggle_mute():
    """Mute if unmuted, unmute if muted. Returns the new state str."""
    if not shutil.which("pactl"):
        print("[system_tools] pactl not found; cannot control audio")
        return "unknown"

    muted = _is_muted()
    if muted is None:
        # Couldn't read state; fall back to a blind toggle.
        _run(["pactl", "set-sink-mute", _DEFAULT_SINK, "toggle"])
        return "toggled"

    target = "0" if muted else "1"  # if currently muted -> unmute, else mute
    _run(["pactl", "set-sink-mute", _DEFAULT_SINK, target])
    return "unmuted" if muted else "muted"
