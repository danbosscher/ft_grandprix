import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import median

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ft_grandprix.custom import Mujoco  # noqa: E402
from scripts.benchmark_driver import HeadlessView, build_cars  # noqa: E402


DEFAULT_DRIVERS = [
    "drivers.daboss_driver",
    "drivers.daboss_mega_driver",
    "drivers.daboss_ultra_driver",
    "drivers.daboss_attack_driver",
    "drivers.daboss_overdrive_driver",
]


@dataclass
class ProgressSample:
    sim_time: float
    absolute_completion: float
    laps: int
    distance_from_track: float
    command_speed: float
    command_steer: float
    actual_speed: float


def sanitize_driver_name(driver_path: str) -> str:
    return driver_path.split(".")[-1]


def sparkline_svg(values, width=420, height=130, stroke="#2563eb", fill="rgba(37,99,235,0.12)"):
    if not values:
        return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"><text x="12" y="24" fill="#6b7280" font-size="14">No data</text></svg>'
    if len(values) == 1:
        value = values[0]
        return (
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
            f'<circle cx="{width/2:.1f}" cy="{height/2:.1f}" r="4" fill="{stroke}" />'
            f'<text x="12" y="24" fill="#111827" font-size="14">{value:.3f}</text>'
            "</svg>"
        )

    min_value = min(values)
    max_value = max(values)
    spread = max(max_value - min_value, 1e-9)
    points = []
    fill_points = [(0, height)]
    for index, value in enumerate(values):
        x = (index / (len(values) - 1)) * width
        y = height - ((value - min_value) / spread) * (height - 12) - 6
        points.append((x, y))
        fill_points.append((x, y))
    fill_points.append((width, height))
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in fill_points)
    y_min = height - ((min_value - min_value) / spread) * (height - 12) - 6
    y_max = height - ((max_value - min_value) / spread) * (height - 12) - 6
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        f'<line x1="0" y1="{y_min:.2f}" x2="{width}" y2="{y_min:.2f}" stroke="#e5e7eb" stroke-width="1" />'
        f'<line x1="0" y1="{y_max:.2f}" x2="{width}" y2="{y_max:.2f}" stroke="#e5e7eb" stroke-width="1" />'
        f'<polygon points="{polygon}" fill="{fill}" />'
        f'<polyline points="{polyline}" fill="none" stroke="{stroke}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        f'<text x="8" y="16" fill="#111827" font-size="12">{max_value:.3f}</text>'
        f'<text x="8" y="{height-8}" fill="#6b7280" font-size="12">{min_value:.3f}</text>'
        "</svg>"
    )


def progress_svg(samples, width=420, height=130):
    if not samples:
        return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"><text x="12" y="24" fill="#6b7280" font-size="14">No data</text></svg>'
    xs = [sample["sim_time"] for sample in samples]
    ys = [sample["absolute_completion"] for sample in samples]
    if len(xs) == 1:
        return f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"><text x="12" y="24" fill="#111827" font-size="14">{ys[0]:.1f}</text></svg>'
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_spread = max(max_x - min_x, 1e-9)
    y_spread = max(max_y - min_y, 1e-9)
    points = []
    for x_value, y_value in zip(xs, ys):
        x = ((x_value - min_x) / x_spread) * width
        y = height - ((y_value - min_y) / y_spread) * (height - 12) - 6
        points.append((x, y))
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        f'<polyline points="{polyline}" fill="none" stroke="#059669" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        f'<text x="8" y="16" fill="#111827" font-size="12">{max_y:.1f} abs</text>'
        f'<text x="8" y="{height-8}" fill="#6b7280" font-size="12">{min_y:.1f} abs</text>'
        "</svg>"
    )


def build_html_report(summary, destination: Path):
    rows = []
    cards = []
    for result in summary["results"]:
        status_class = "ok" if result["status"] == "finished" else "warn"
        rows.append(
            "<tr>"
            f"<td>{escape(result['driver'])}</td>"
            f"<td><span class='status {status_class}'>{escape(result['status'])}</span></td>"
            f"<td>{result['laps']}</td>"
            f"<td>{result['best_lap'] if result['best_lap'] is not None else '-'}</td>"
            f"<td>{result['median_lap'] if result['median_lap'] is not None else '-'}</td>"
            f"<td>{result['last_lap'] if result['last_lap'] is not None else '-'}</td>"
            f"<td>{result['total_time']}</td>"
            f"<td>{result['best_absolute_completion']}</td>"
            "</tr>"
        )
        cards.append(
            "<section class='card'>"
            f"<h2>{escape(result['driver'])}</h2>"
            f"<p class='meta'>status: <strong>{escape(result['status'])}</strong> | laps: <strong>{result['laps']}</strong> | best lap: <strong>{result['best_lap'] if result['best_lap'] is not None else '-'}</strong></p>"
            "<div class='charts'>"
            "<div>"
            "<h3>Lap Times</h3>"
            f"{sparkline_svg(result['lap_times'])}"
            "</div>"
            "<div>"
            "<h3>Absolute Completion</h3>"
            f"{progress_svg(result['progress_samples'])}"
            "</div>"
            "</div>"
            "<details><summary>Raw Summary</summary>"
            f"<pre>{escape(json.dumps(result, indent=2))}</pre>"
            "</details>"
            "</section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FT Grand Prix Long Benchmark</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffdf9;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d6d3d1;
      --ok: #065f46;
      --warn: #92400e;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top, #fdf7e7 0%, var(--bg) 48%, #ece8df 100%);
      color: var(--ink);
      font: 16px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 60px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 38px;
      line-height: 1.1;
    }}
    .lede {{
      color: var(--muted);
      max-width: 760px;
      margin-bottom: 28px;
    }}
    .summary {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
      margin-bottom: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 10px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .status.ok {{
      background: rgba(16, 185, 129, 0.12);
      color: var(--ok);
    }}
    .status.warn {{
      background: rgba(245, 158, 11, 0.14);
      color: var(--warn);
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 20px 50px rgba(15, 23, 42, 0.06);
      margin-bottom: 24px;
    }}
    .card h2 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    .meta {{
      color: var(--muted);
      margin: 0 0 18px;
    }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .charts h3 {{
      margin: 0 0 8px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--accent);
    }}
    details {{
      margin-top: 16px;
    }}
    pre {{
      overflow-x: auto;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 14px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>FT Grand Prix 100-Lap Report</h1>
    <p class="lede">Generated at {escape(summary['generated_at'])}. Track: <strong>{escape(summary['track'])}</strong>. Lap target: <strong>{summary['lap_target']}</strong>. The report includes raw benchmark logs, lap-time graphs, and absolute-completion traces for each custom driver.</p>
    <section class="summary">
      <h2>Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Driver</th>
            <th>Status</th>
            <th>Laps</th>
            <th>Best Lap</th>
            <th>Median Lap</th>
            <th>Last Lap</th>
            <th>Total Time</th>
            <th>Best Absolute Completion</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    destination.write_text(html, encoding="utf-8")


def write_driver_logs(driver_result, output_dir: Path):
    stem = sanitize_driver_name(driver_result["driver"])
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}_laps.csv"
    json_path.write_text(json.dumps(driver_result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lap_index", "lap_time_seconds"])
        for index, lap_time in enumerate(driver_result["lap_times"], start=1):
            writer.writerow([index, lap_time])


def run_driver(driver_path: str, lap_target: int, track: str, physics_fps: int, wall_timeout: float, no_progress_sim_seconds: float, progress_stride_steps: int, max_lap_seconds: float | None):
    view = HeadlessView()
    mj = Mujoco(view, track=track)
    mj.cars = build_cars(driver_path)
    mj.nuke("cars_path")
    mj.option("save_on_exit", False)
    mj.option("lap_target", lap_target)
    mj.option("max_fps", 1)
    mj.option("physics_fps", physics_fps)
    mj.restart_render_thread = lambda: None
    mj.stage()
    mj.option("bubble_wrap", False)

    vehicle_state = mj.vehicle_states[0]
    best_absolute_completion = float(vehicle_state.absolute_completion())
    last_progress_steps = 0
    timestep = mj.model.opt.timestep
    start = time.time()
    status = "timeout"
    stall_reason = None
    progress_samples = []
    last_logged_lap_count = 0

    while time.time() < start + wall_timeout:
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
                lap_time = (mj.steps - vehicle_state.start) * timestep
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
                status = "finished"
                break
            vehicle_state.completion = completion
            absolute_completion = float(vehicle_state.absolute_completion())
            if absolute_completion > best_absolute_completion:
                best_absolute_completion = absolute_completion
                last_progress_steps = mj.steps

        ranges = mj.data.sensordata[vehicle_state.sensors]
        snapshot = vehicle_state.snapshot(time=mj.steps / timestep)
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

        if mj.steps % progress_stride_steps == 0:
            progress_samples.append(
                asdict(
                    ProgressSample(
                        sim_time=float(mj.steps * timestep),
                        absolute_completion=float(vehicle_state.absolute_completion()),
                        laps=int(vehicle_state.laps),
                        distance_from_track=float(vehicle_state.distance_from_track),
                        command_speed=float(vehicle_state.speed),
                        command_steer=float(vehicle_state.steering_angle),
                        actual_speed=float(np.linalg.norm(np.asarray(vehicle_state.joint.qvel[:2]))),
                    )
                )
            )

        if len(vehicle_state.times) > last_logged_lap_count:
            last_logged_lap_count = len(vehicle_state.times)

        no_progress_sim = (mj.steps - last_progress_steps) * timestep
        if no_progress_sim > no_progress_sim_seconds:
            status = "stalled"
            stall_reason = f"no progress for {no_progress_sim:.1f}s sim"
            break

        if max_lap_seconds is not None:
            current_lap_time = (mj.steps - vehicle_state.start) * timestep
            if current_lap_time > max_lap_seconds:
                status = "lap_timeout"
                stall_reason = f"current lap exceeded {max_lap_seconds:.1f}s sim"
                break

    result = {
        "driver": driver_path,
        "status": status,
        "laps": int(vehicle_state.laps),
        "finished": bool(vehicle_state.finished),
        "lap_times": [round(float(value), 3) for value in vehicle_state.times],
        "total_time": round(float(sum(vehicle_state.times)), 3),
        "best_lap": round(float(min(vehicle_state.times)), 3) if vehicle_state.times else None,
        "median_lap": round(float(median(vehicle_state.times)), 3) if vehicle_state.times else None,
        "last_lap": round(float(vehicle_state.times[-1]), 3) if vehicle_state.times else None,
        "absolute_completion": round(float(vehicle_state.absolute_completion()), 3),
        "best_absolute_completion": round(float(best_absolute_completion), 3),
        "distance_from_track": round(float(vehicle_state.distance_from_track), 6),
        "final_speed": round(float(np.linalg.norm(np.asarray(vehicle_state.joint.qvel[:2]))), 6),
        "final_command": {
            "speed": round(float(vehicle_state.speed), 6),
            "steering": round(float(vehicle_state.steering_angle), 6),
        },
        "steps": int(mj.steps),
        "sim_time": round(float(mj.steps * timestep), 3),
        "wall_time": round(time.time() - start, 3),
        "stall_reason": stall_reason,
        "progress_samples": progress_samples,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laps", type=int, default=100)
    parser.add_argument("--track", default="track")
    parser.add_argument("--physics-fps", type=int, default=500)
    parser.add_argument("--wall-timeout", type=float, default=1800.0)
    parser.add_argument("--no-progress-sim-seconds", type=float, default=45.0)
    parser.add_argument("--progress-stride-steps", type=int, default=250)
    parser.add_argument("--max-lap-seconds", type=float, default=None)
    parser.add_argument("--output-root", default="reports")
    parser.add_argument("--driver", action="append", dest="drivers")
    args = parser.parse_args()

    drivers = args.drivers or list(DEFAULT_DRIVERS)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = REPO_ROOT / args.output_root / f"long-benchmark-{timestamp}"
    logs_dir = output_dir / "drivers"
    logs_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "events.jsonl"
    results = []
    for driver_path in drivers:
        start_event = {"event": "start", "driver": driver_path, "time": time.time()}
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(start_event) + "\n")
        print(json.dumps(start_event), flush=True)

        result = run_driver(
            driver_path=driver_path,
            lap_target=args.laps,
            track=args.track,
            physics_fps=args.physics_fps,
            wall_timeout=args.wall_timeout,
            no_progress_sim_seconds=args.no_progress_sim_seconds,
            progress_stride_steps=args.progress_stride_steps,
            max_lap_seconds=args.max_lap_seconds,
        )
        results.append(result)
        write_driver_logs(result, logs_dir)
        finish_event = {
            "event": "finish",
            "driver": driver_path,
            "status": result["status"],
            "laps": result["laps"],
            "wall_time": result["wall_time"],
            "time": time.time(),
        }
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(finish_event) + "\n")
        print(json.dumps(result), flush=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track": args.track,
        "lap_target": args.laps,
        "physics_fps": args.physics_fps,
        "wall_timeout": args.wall_timeout,
        "no_progress_sim_seconds": args.no_progress_sim_seconds,
        "max_lap_seconds": args.max_lap_seconds,
        "drivers": drivers,
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_html_report(summary, output_dir / "report.html")
    print(json.dumps({"event": "report_ready", "output_dir": str(output_dir), "report_html": str(output_dir / "report.html")}), flush=True)


if __name__ == "__main__":
    main()
