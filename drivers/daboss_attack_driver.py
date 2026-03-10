import json
import os

import numpy as np


DEFAULT_PARAMS = {
    "car_width": 0.06,
    "difference_threshold": 0.6,
    "safety_percentage": 300.0,
    "straight_speed": 7.5,
    "straight_angle_threshold": 0.1,
    "straight_distance_threshold": 0.5,
    "corner_speed_scale": 3.3,
    "corner_speed_cap": 3.0,
    "corner_power": 1.0,
    "rear_ignore_fraction": 0.125,
}


class Driver:
    def __init__(self):
        params = dict(DEFAULT_PARAMS)
        override_blob = os.environ.get("FTGP_DRIVER_PARAMS")
        if override_blob:
            params.update(json.loads(override_blob))
        self.params = params
        self.last_steering_angle = 0.0

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

    def corner_speed(self, steering_angle):
        normalized = max(0.0, 1 - abs(steering_angle) / np.pi)
        scaled = self.params["corner_speed_scale"] * normalized ** self.params["corner_power"]
        return min(self.params["corner_speed_cap"], scaled)

    def process_lidar(self, ranges):
        self.radians_per_point = (2 * np.pi) / len(ranges)
        processed = self.preprocess_lidar(ranges)
        differences = self.get_differences(processed)
        disparities = self.get_disparities(differences)
        processed = self.extend_disparities(disparities, processed)
        steering_angle = self.get_steering_angle(int(processed.argmax()), len(processed))
        self.last_steering_angle = steering_angle

        if abs(steering_angle) < self.params["straight_angle_threshold"] and ranges[0] > self.params["straight_distance_threshold"]:
            speed = self.params["straight_speed"]
        else:
            speed = self.corner_speed(steering_angle)
        return speed, steering_angle
