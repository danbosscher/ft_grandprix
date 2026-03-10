import json
import math
import os

import numpy as np


STEER_PROFILE = [
    0.139626, 0.0, -0.069813, 0.069813, 0.0, 0.0, 0.0, 0.0, -0.069813, -0.139626,
    -0.767945, -0.837758, -0.907571, -0.628319, 0.069813, 0.0, -0.069813, -0.069813, -0.20944, 0.0,
    -1.047198, -0.837758, 0.20944, 0.20944, 0.0, 0.069813, -0.069813, 0.069813, 0.0, 0.069813,
    0.0, 0.0, 0.0, 0.069813, 0.069813, 0.20944, 0.767945, -0.349066, 0.069813, 0.0,
    -0.069813, -0.069813, -0.418879, -0.488692, -0.418879, -0.20944, -0.069813, -0.139626, -0.139626, 0.0,
    -0.698132, -0.558505, 0.139626, 0.069813, 0.069813, 0.069813, 0.069813, 0.0, 0.558505, 0.0,
    -0.139626, -0.069813, -0.069813, 0.0, -0.139626, -0.558505, -0.20944, 0.0, -0.069813, 0.0,
    -0.418879, -0.20944, 0.0, -0.069813, 0.0, -0.139626, -0.279253, -0.034907, 0.20944, 0.20944,
    0.20944, 0.349066, 0.20944, 0.034907, 0.069813, 0.069813, 0.0, -0.069813, -0.069813, -0.558505,
    -0.279253, 0.139626, -0.069813, -0.069813, 0.069813, 0.0, 0.488692, 0.488692, 0.837758, 0.628319,
]

YAW_PROFILE = [
    -0.135101, -0.127555, -0.124292, -0.12375, -0.125412, -0.129318, -0.136022, -0.146868, -0.164912, -0.198098,
    -0.274221, -0.73891, -1.25938, -1.419516, -1.493907, -1.546122, -1.597405, -1.665734, -1.792474, -2.280995,
    -3.080504, 3.018276, 2.940641, 2.88435, 2.843122, 2.813075, 2.791842, 2.778089, 2.771273, 2.771611,
    2.780255, 2.79983, 2.835771, 2.899835, 3.020608, -3.004082, -2.290864, -1.875916, -1.652403, -1.561391,
    -1.542136, -1.586357, -1.737663, -2.13917, -2.47874, -2.637888, -2.762704, -2.918044, 3.063219, 2.340479,
    1.721123, 1.560271, 1.503817, 1.49198, 1.516137, 1.597624, 1.877002, 2.75843, -3.000884, -2.853283,
    -2.81843, -2.851026, -2.986669, 2.848198, 2.302429, 2.125655, 2.025644, 1.930942, 1.799687, 1.532558,
    1.13058, 0.950516, 0.865496, 0.801977, 0.735405, 0.636383, 0.366477, -0.167079, -0.401415, -0.474072,
    -0.452467, -0.135828, 0.485542, 0.736048, 0.84686, 0.909551, 0.973092, 0.994357, 0.875436, 0.523001,
    -0.191278, -0.940833, -1.492214, -1.517741, -1.492997, -1.418689, -1.224837, -0.769389, -0.216591, -0.151455,
]

DEFAULT_PARAMS = {
    "car_width": 0.06,
    "difference_threshold": 0.6,
    "safety_percentage": 300.0,
    "straight_speed": 7.0,
    "boosted_straight_speed": 7.4,
    "straight_angle_threshold": 0.1,
    "straight_distance_threshold": 0.5,
    "corner_speed_scale": 3.15,
    "corner_speed_cap": 2.9,
    "corner_power": 1.0,
    "rear_ignore_fraction": 0.125,
    "bucket_count": 20,
    "boosted_buckets": [],
    "bucket_speed_overrides": {11: 7.4},
    "recovery_enable_after_completion": 100.0,
    "stall_speed_threshold": 0.1,
    "stall_front_threshold": 0.22,
    "stall_trigger_steps": 35,
    "regression_margin": 2.0,
    "regression_trigger_steps": 5,
    "heading_recovery_trigger": 2.4,
    "reverse_steps": 10,
    "recovery_steps": 24,
    "reverse_speed": -1.2,
    "recovery_speed": 2.6,
    "recovery_steer": 1.05,
}


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Driver:
    def __init__(self):
        params = dict(DEFAULT_PARAMS)
        override_blob = os.environ.get("FTGP_DRIVER_PARAMS")
        if override_blob:
            params.update(json.loads(override_blob))
        params["boosted_buckets"] = set(params["boosted_buckets"])
        params["bucket_speed_overrides"] = {
            int(bucket): float(speed)
            for bucket, speed in params["bucket_speed_overrides"].items()
        }
        self.params = params
        self.last_steering_angle = 0.0
        self.best_absolute_completion = -1e9
        self.previous_absolute_completion = None
        self.stall_steps = 0
        self.regression_steps = 0
        self.reverse_steps_remaining = 0
        self.recovery_steps_remaining = 0
        self.recovery_direction = 1.0

    def preprocess_lidar(self, ranges):
        ignore = int(len(ranges) * self.params["rear_ignore_fraction"])
        return np.array(ranges[ignore:-ignore])

    def get_differences(self, ranges):
        differences = [0.0]
        for i in range(1, len(ranges)):
            differences.append(abs(ranges[i] - ranges[i - 1]))
        return differences

    def get_disparities(self, differences):
        return [index for index, difference in enumerate(differences) if difference > self.params["difference_threshold"]]

    def get_num_points_to_cover(self, dist, width):
        angle = 2 * np.arctan(width / (2 * dist))
        return int(np.ceil(angle / self.radians_per_point))

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        new_dist = ranges[start_idx]
        if cover_right:
            for i in range(num_points):
                next_idx = start_idx + 1 + i
                if next_idx >= len(ranges):
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        else:
            for i in range(num_points):
                next_idx = start_idx - 1 - i
                if next_idx < 0:
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        return ranges

    def extend_disparities(self, disparities, ranges):
        width_to_cover = (self.params["car_width"] / 2) * (1 + self.params["safety_percentage"] / 100)
        for index in disparities:
            first_idx = index - 1
            points = ranges[first_idx:first_idx + 2]
            close_idx = first_idx + np.argmin(points)
            far_idx = first_idx + np.argmax(points)
            close_dist = ranges[close_idx]
            num_points_to_cover = self.get_num_points_to_cover(close_dist, width_to_cover)
            cover_right = close_idx < far_idx
            ranges = self.cover_points(num_points_to_cover, close_idx, cover_right, ranges)
        return ranges

    def get_steering_angle(self, range_index, range_len):
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_point
        return np.clip(lidar_angle, np.radians(-90), np.radians(90))

    def straight_speed(self, state):
        bucket_size = 100.0 / int(self.params["bucket_count"])
        bucket = min(int(self.params["bucket_count"]) - 1, max(0, int(state.lap_completion // bucket_size)))
        if bucket in self.params["bucket_speed_overrides"]:
            return self.params["bucket_speed_overrides"][bucket]
        if bucket in self.params["boosted_buckets"]:
            return self.params["boosted_straight_speed"]
        return self.params["straight_speed"]

    def corner_speed(self, steering_angle):
        normalized = max(0.0, 1 - abs(steering_angle) / np.pi)
        scaled = self.params["corner_speed_scale"] * normalized ** self.params["corner_power"]
        return min(self.params["corner_speed_cap"], scaled)

    def update_progress(self, state, front_clear, actual_speed):
        absolute_completion = float(state.absolute_completion)
        if absolute_completion > self.best_absolute_completion:
            self.best_absolute_completion = absolute_completion
        if self.previous_absolute_completion is not None:
            delta = absolute_completion - self.previous_absolute_completion
            if delta < -0.1 or absolute_completion < self.best_absolute_completion - self.params["regression_margin"]:
                self.regression_steps += 1
            elif delta > 0.05:
                self.regression_steps = max(0, self.regression_steps - 2)
            else:
                self.regression_steps = max(0, self.regression_steps - 1)
        self.previous_absolute_completion = absolute_completion

        if actual_speed < self.params["stall_speed_threshold"] and front_clear < self.params["stall_front_threshold"]:
            self.stall_steps += 1
        else:
            self.stall_steps = max(0, self.stall_steps - 1)

    def start_recovery(self, completion, heading_error):
        hint = STEER_PROFILE[completion]
        if abs(hint) > 0.1:
            self.recovery_direction = 1.0 if hint >= 0.0 else -1.0
        elif abs(self.last_steering_angle) > 0.05:
            self.recovery_direction = 1.0 if self.last_steering_angle >= 0.0 else -1.0
        else:
            self.recovery_direction = 1.0 if heading_error >= 0.0 else -1.0
        self.reverse_steps_remaining = int(self.params["reverse_steps"])
        self.recovery_steps_remaining = int(self.params["recovery_steps"])
        self.stall_steps = 0
        self.regression_steps = 0

    def maybe_recover(self, completion, heading_error, front_clear, actual_speed, state):
        if self.reverse_steps_remaining > 0:
            self.reverse_steps_remaining -= 1
            return self.params["reverse_speed"], 0.0

        if self.recovery_steps_remaining > 0:
            self.recovery_steps_remaining -= 1
            if abs(heading_error) < 0.4 and actual_speed > 0.8:
                self.recovery_steps_remaining = 0
            return self.params["recovery_speed"], self.recovery_direction * self.params["recovery_steer"]

        if self.best_absolute_completion < self.params["recovery_enable_after_completion"]:
            return None

        should_recover = (
            self.stall_steps >= self.params["stall_trigger_steps"]
            or self.regression_steps >= self.params["regression_trigger_steps"]
            or (abs(heading_error) > self.params["heading_recovery_trigger"] and front_clear < 0.35 and actual_speed < 0.4)
        )
        if should_recover:
            self.start_recovery(completion, heading_error)
            return self.maybe_recover(completion, heading_error, front_clear, actual_speed, state)
        return None

    def process_lidar(self, ranges, state):
        self.radians_per_point = (2 * np.pi) / len(ranges)
        processed = self.preprocess_lidar(ranges)
        differences = self.get_differences(processed)
        disparities = self.get_disparities(differences)
        processed = self.extend_disparities(disparities, processed)
        steering_angle = self.get_steering_angle(int(processed.argmax()), len(processed))

        if abs(steering_angle) < self.params["straight_angle_threshold"] and ranges[0] > self.params["straight_distance_threshold"]:
            speed = self.straight_speed(state)
        else:
            speed = self.corner_speed(steering_angle)

        front_clear = float(ranges[0])
        completion = int(np.clip(state.lap_completion, 0, 99))
        heading_error = wrap_angle(float(YAW_PROFILE[completion] - state.yaw))
        actual_speed = float(np.linalg.norm(np.asarray(state.velocity[:2], dtype=float)))
        self.update_progress(state, front_clear, actual_speed)

        recovery = self.maybe_recover(completion, heading_error, front_clear, actual_speed, state)
        if recovery is not None:
            speed, steering_angle = recovery
            self.last_steering_angle = float(steering_angle)
            return float(speed), float(steering_angle)

        self.last_steering_angle = steering_angle
        return float(speed), float(steering_angle)
