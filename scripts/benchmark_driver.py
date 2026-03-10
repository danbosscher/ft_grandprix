import argparse
import json
import sys
import time
from pathlib import Path
from threading import Event

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ft_grandprix.custom import Mujoco  # noqa: E402


class HeadlessView:
    def __init__(self):
        self.viewport_resize_event = Event()
        self.speed = 0.0
        self.steering_angle = 0.0
        self.pixels = np.zeros((1, 1, 3), dtype=np.float32)

    def simulation_viewport_size(self):
        return 0, 0


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


def step_race(mj: Mujoco, deadline: float):
    vehicle_state = mj.vehicle_states[0]
    best_absolute_completion = float(vehicle_state.absolute_completion())
    while time.time() < deadline:
        xpos = vehicle_state.joint.qpos[0:2]
        distances = ((mj.path - xpos) ** 2).sum(1)
        closest = int(distances.argmin())
        vehicle_state.distance_from_track = float(distances[closest])
        vehicle_state.off_track = bool(vehicle_state.distance_from_track > 1.0)
        if not vehicle_state.off_track:
            completion = (closest - vehicle_state.offset) % 100
            delta = completion - vehicle_state.completion
            vehicle_state.delta = (completion - vehicle_state.completion + 50) % 100 - 50
            if abs(delta) > 90:
                lap_time = (mj.steps - vehicle_state.start) * mj.model.opt.timestep
                if vehicle_state.delta < 0:
                    vehicle_state.laps -= 1
                    vehicle_state.good_start = False
                    if vehicle_state.times:
                        vehicle_state.times.pop()
                elif vehicle_state.delta > 0:
                    if vehicle_state.good_start:
                        vehicle_state.times.append(lap_time)
                        vehicle_state.start = mj.steps
                    vehicle_state.laps += 1
                    vehicle_state.good_start = True
            if vehicle_state.laps >= mj.option("lap_target"):
                vehicle_state.finished = True
                break
            vehicle_state.completion = completion
            best_absolute_completion = max(best_absolute_completion, float(vehicle_state.absolute_completion()))

        ranges = mj.data.sensordata[vehicle_state.sensors]
        snapshot = vehicle_state.snapshot(time=mj.steps / mj.model.opt.timestep)
        if vehicle_state.v2:
            speed, steering_angle = vehicle_state.driver.process_lidar(ranges, snapshot)
        else:
            speed, steering_angle = vehicle_state.driver.process_lidar(ranges)

        vehicle_state.speed = speed
        vehicle_state.steering_angle = steering_angle
        mj.data.ctrl[vehicle_state.forward] = speed
        mj.data.ctrl[vehicle_state.turn] = steering_angle
        mujoco.mj_step(mj.model, mj.data)
        mj.steps += 1
    return best_absolute_completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="drivers.daboss_driver")
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--track", default="track")
    parser.add_argument("--physics-fps", type=int, default=500)
    parser.add_argument("--params", default=None, help="JSON object merged into FTGP_DRIVER_PARAMS")
    args = parser.parse_args()

    if args.params:
        payload = json.loads(args.params)
        os_payload = json.dumps(payload, separators=(",", ":"))
        import os

        os.environ["FTGP_DRIVER_PARAMS"] = os_payload

    view = HeadlessView()
    mj = Mujoco(view, track=args.track)
    mj.cars = build_cars(args.driver)
    mj.nuke("cars_path")
    mj.option("save_on_exit", False)
    mj.option("lap_target", args.laps)
    mj.option("max_fps", 1)
    mj.option("physics_fps", args.physics_fps)
    mj.restart_render_thread = lambda: None

    mj.stage()
    mj.option("bubble_wrap", False)

    start = time.time()
    best_absolute_completion = step_race(mj, start + args.timeout)
    vehicle_state = mj.vehicle_states[0]
    result = {
        "driver": args.driver,
        "finished": bool(vehicle_state.finished),
        "laps": int(vehicle_state.laps),
        "lap_times": [float(value) for value in vehicle_state.times],
        "total_time": float(sum(vehicle_state.times)),
        "best_lap": float(min(vehicle_state.times)) if vehicle_state.times else None,
        "completion": float(vehicle_state.lap_completion()),
        "absolute_completion": float(vehicle_state.absolute_completion()),
        "best_absolute_completion": float(best_absolute_completion),
        "distance_from_track": float(vehicle_state.distance_from_track),
        "final_speed": float(np.linalg.norm(np.asarray(vehicle_state.joint.qvel[:2]))),
        "final_command": {
            "speed": float(vehicle_state.speed),
            "steering": float(vehicle_state.steering_angle),
        },
        "steps": int(mj.steps),
        "sim_time": float(mj.steps * mj.model.opt.timestep),
        "wall_time": time.time() - start,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
