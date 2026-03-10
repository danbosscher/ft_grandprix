import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ft_grandprix.custom import Command, Event, ModelAndView, Mujoco, platform  # noqa: E402


def build_cars(driver_path: str):
    return [
        {
            "driver": driver_path,
            "name": driver_path,
            "primary": "white",
            "secondary": "deepskyblue",
            "icon": "white.png",
        }
    ]


def build_view(track: str, driver_path: str, width: int = 1124, height: int = 612):
    mv = ModelAndView.__new__(ModelAndView)
    print("Running on", platform)
    mv.viewport_resize_event = Event()
    mv.window_size = (width, height)
    if platform == "darwin":
        mv.pixels = np.zeros((1080, 1920, 4), dtype=np.float32)
        mv.pixels[..., 3] = 1.0
    else:
        mv.pixels = np.zeros((1080, 1920, 3), dtype=np.float32)
    mv.last = None
    mv.speed = 0.0
    mv.steering_angle = 0.0
    mv.mj = Mujoco(mv, track=track)
    mv.mj.nuke("cars_path")
    mv.mj.cars = build_cars(driver_path)
    mv.mj.run()
    mv.mj.option("bubble_wrap", False)
    mv.mj.watching = 0

    mv.commands = {
        command.tag: command
        for command in [
            Command("pause", mv.pause),
            Command("reset", mv.reset),
            Command("hard_reset", mv.hard_reset),
            Command("zoom_in", lambda: mv.scroll_cb(None, 10)),
            Command("zoom_out", lambda: mv.scroll_cb(None, -10)),
            Command("focus_on_next_car", mv.focus_on_next_car),
            Command("focus_on_previous_car", mv.focus_on_previous_car),
            Command(
                "reload_code",
                lambda: mv.reload_code_cb(None, None, mv.mj.watching),
                description="Reloads the code for the vehicle currently focused on",
            ),
            Command("toggle_cinematic_camera", mv.toggle_cinematic_camera),
            Command(
                "liberate_camera",
                lambda: setattr(mv.mj, "watching", None),
                description="Stops focusing on any car",
            ),
            Command(
                "reload_code",
                lambda: mv.reload_code_cb(None, None, mv.mj.watching),
                description="Reloads the code for the that is currently being watched",
            ),
            Command("toggle_shadows", mv.toggle_shadows),
            Command("rotate_camera_left", lambda: mv.mj.perturb_camera(-5, 0)),
            Command("rotate_camera_right", lambda: mv.mj.perturb_camera(5, 0)),
            Command("rotate_camera_up", lambda: mv.mj.perturb_camera(0, -5)),
            Command("rotate_camera_down", lambda: mv.mj.perturb_camera(0, 5)),
            Command("show_cars_modal", mv.show_cars_modal),
            Command("show_keybindings_modal", mv.show_keybindings_modal),
            Command("sync_options", mv.mj.sync_options),
            Command("position_vehicles", mv.mj.position_vehicles),
        ]
    }

    mv.keybindings = {
        " ": "pause",
        "E": "reset",
        "R": "reload_code",
        "-": "zoom_out",
        "+": "zoom_in",
        "N": "focus_on_next_car",
        "P": "focus_on_previous_car",
        "C": "toggle_cinematic_camera",
        "L": "liberate_camera",
        "H": "toggle_shadows",
        263: "rotate_camera_left",
        262: "rotate_camera_right",
        265: "rotate_camera_up",
        264: "rotate_camera_down",
    }
    return mv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="drivers.daboss_driver")
    parser.add_argument("--track", default="track")
    args = parser.parse_args()

    mv = build_view(args.track, args.driver)
    mv.mj.running_event.set()
    mv.run()


if __name__ == "__main__":
    main()
