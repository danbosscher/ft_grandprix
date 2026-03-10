import json
import os

import numpy as np


DEFAULT_PARAMS = {
    "car_width": 0.06,
    "difference_threshold": 0.6,
    "safety_percentage": 300.0,
    "straight_speed": 7.2,
    "straight_angle_threshold": 0.1,
    "straight_distance_threshold": 0.5,
    "corner_speed_scale": 3.25,
    "corner_speed_cap": 3.0,
    "corner_power": 1.0,
    "front_speed_gain": 4.35,
    "front_speed_floor": 1.45,
    "rear_ignore_fraction": 0.125,
    "center_bias": 0.0,
    "steering_alpha": 1.0,
    "max_steer_delta": 10.0,
    "stall_front_threshold": 0.18,
    "stall_trigger_steps": 28,
    "stuck_variation_threshold": 0.008,
    "reverse_steps": 8,
    "recovery_steps": 12,
    "reverse_speed": -1.2,
    "reverse_steer": 0.95,
    "recovery_speed": 2.4,
    "recovery_steer": 0.9,
    "max_recovery_cycles": 2,
    "speed_scale_drop_per_recovery": 0.02,
    "speed_scale_floor": 0.86,
}


class Driver:
    def __init__(self):
        params = dict(DEFAULT_PARAMS)
        override_blob = os.environ.get("FTGP_DRIVER_PARAMS")
        if override_blob:
            params.update(json.loads(override_blob))
        self.params = params
        self.last_steering_angle = 0.0
        self.last_front_clear = None
        self.stall_steps = 0
        self.reverse_steps_remaining = 0
        self.recovery_steps_remaining = 0
        self.recovery_direction = 1.0
        self.recovery_cycles = 0
        self.total_recoveries = 0

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

    def pick_target_index(self, processed):
        center = (len(processed) - 1) / 2.0
        offsets = np.abs(np.arange(len(processed)) - center) / max(center, 1.0)
        weighted = processed * (1.0 - self.params["center_bias"] * offsets)
        return int(np.argmax(weighted))

    def corner_speed(self, steering_angle):
        normalized = max(0.0, 1 - abs(steering_angle) / np.pi)
        scaled = self.params["corner_speed_scale"] * normalized ** self.params["corner_power"]
        return min(self.params["corner_speed_cap"], scaled)

    def speed_scale(self):
        return max(
            self.params["speed_scale_floor"],
            1.0 - self.total_recoveries * self.params["speed_scale_drop_per_recovery"],
        )

    def front_clearance(self, ranges, processed):
        processed_front = float(processed[len(processed) // 2])
        raw_front = float(ranges[0])
        return max(raw_front, processed_front), min(raw_front, processed_front)

    def update_stall_state(self, front_tight, steering_angle):
        if self.last_front_clear is None:
            front_delta = 1.0
        else:
            front_delta = abs(front_tight - self.last_front_clear)
        self.last_front_clear = front_tight

        low_front = front_tight < self.params["stall_front_threshold"]
        low_variation = front_delta < self.params["stuck_variation_threshold"]
        if low_front and (low_variation or abs(steering_angle) > 0.55):
            self.stall_steps += 1
        else:
            self.stall_steps = max(0, self.stall_steps - 1)

    def choose_recovery_direction(self, processed):
        midpoint = len(processed) // 2
        left_open = float(np.mean(processed[:midpoint]))
        right_open = float(np.mean(processed[midpoint:]))
        if abs(left_open - right_open) > 0.02:
            return -1.0 if left_open > right_open else 1.0
        if abs(self.last_steering_angle) > 0.05:
            return 1.0 if self.last_steering_angle >= 0.0 else -1.0
        return -1.0

    def start_recovery(self, processed):
        self.recovery_direction = self.choose_recovery_direction(processed)
        self.reverse_steps_remaining = int(self.params["reverse_steps"])
        self.recovery_steps_remaining = int(self.params["recovery_steps"])
        self.recovery_cycles = 1
        self.stall_steps = 0
        self.total_recoveries += 1

    def maybe_recover(self, processed, front_clear, front_tight):
        if self.reverse_steps_remaining > 0:
            self.reverse_steps_remaining -= 1
            return self.params["reverse_speed"], -self.recovery_direction * self.params["reverse_steer"]

        if self.recovery_steps_remaining > 0:
            self.recovery_steps_remaining -= 1
            if front_clear > 0.55 and front_tight > self.params["stall_front_threshold"]:
                self.recovery_steps_remaining = 0
                self.recovery_cycles = 0
                return None
            return self.params["recovery_speed"], self.recovery_direction * self.params["recovery_steer"]

        if self.recovery_cycles > 0 and front_clear < 0.4:
            if self.recovery_cycles >= self.params["max_recovery_cycles"]:
                self.recovery_cycles = 0
                return None
            self.recovery_cycles += 1
            self.recovery_direction *= -1.0
            self.reverse_steps_remaining = int(self.params["reverse_steps"])
            self.recovery_steps_remaining = int(self.params["recovery_steps"])
            return self.maybe_recover(processed, front_clear, front_tight)

        self.recovery_cycles = 0
        if self.stall_steps >= self.params["stall_trigger_steps"]:
            self.start_recovery(processed)
            return self.maybe_recover(processed, front_clear, front_tight)
        return None

    def smooth_steering(self, steering_angle):
        blended = (
            self.params["steering_alpha"] * steering_angle
            + (1.0 - self.params["steering_alpha"]) * self.last_steering_angle
        )
        limited = np.clip(
            blended,
            self.last_steering_angle - self.params["max_steer_delta"],
            self.last_steering_angle + self.params["max_steer_delta"],
        )
        return float(limited)

    def process_lidar(self, ranges):
        self.radians_per_point = (2 * np.pi) / len(ranges)
        processed = self.preprocess_lidar(ranges)
        differences = self.get_differences(processed)
        disparities = self.get_disparities(differences)
        processed = self.extend_disparities(disparities, processed)

        target_index = self.pick_target_index(processed)
        steering_angle = self.get_steering_angle(target_index, len(processed))
        steering_angle = self.smooth_steering(steering_angle)

        front_clear, front_tight = self.front_clearance(ranges, processed)
        self.update_stall_state(front_tight, steering_angle)

        if abs(steering_angle) < self.params["straight_angle_threshold"] and front_clear > self.params["straight_distance_threshold"]:
            speed = self.params["straight_speed"]
        else:
            speed = self.corner_speed(steering_angle)
        speed = min(speed, max(self.params["front_speed_floor"], self.params["front_speed_gain"] * front_clear))
        speed *= self.speed_scale()

        recovery = self.maybe_recover(processed, front_clear, front_tight)
        if recovery is not None:
            speed, steering_angle = recovery

        self.last_steering_angle = float(steering_angle)
        return float(speed), float(steering_angle)
