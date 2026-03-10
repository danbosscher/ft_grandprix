import argparse
import csv
import json
import math
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


@dataclass
class TelemetrySample:
    step: int
    sim_time: float
    lap_completion: float
    absolute_completion: float
    distance_from_track: float
    command_speed: float
    command_steer: float
    actual_speed: float
    pos_x: float
    pos_y: float
    yaw: float


def make_sample(step: int, timestep: float, vehicle_state, yaw: float) -> dict:
    velocity = np.asarray(vehicle_state.joint.qvel[:2], dtype=float)
    return asdict(
        TelemetrySample(
            step=int(step),
            sim_time=round(float(step * timestep), 6),
            lap_completion=round(float(vehicle_state.lap_completion()), 6),
            absolute_completion=round(float(vehicle_state.absolute_completion()), 6),
            distance_from_track=round(float(vehicle_state.distance_from_track), 9),
            command_speed=round(float(vehicle_state.speed), 6),
            command_steer=round(float(vehicle_state.steering_angle), 6),
            actual_speed=round(float(np.linalg.norm(velocity)), 6),
            pos_x=round(float(vehicle_state.joint.qpos[0]), 6),
            pos_y=round(float(vehicle_state.joint.qpos[1]), 6),
            yaw=round(float(yaw), 6),
        )
    )


def compute_lap_reports(samples: list[dict], lap_times: list[float]) -> list[dict]:
    reports = []
    elapsed_so_far = 0.0
    sample_cursor = 0

    for lap_index, lap_time in enumerate(lap_times, start=1):
        lap_start = elapsed_so_far
        lap_end = elapsed_so_far + float(lap_time)

        lap_samples = []
        while sample_cursor < len(samples) and samples[sample_cursor]["sim_time"] < lap_start:
            sample_cursor += 1
        cursor = sample_cursor
        while cursor < len(samples) and samples[cursor]["sim_time"] <= lap_end + 1e-9:
            sample = dict(samples[cursor])
            sample["elapsed"] = round(float(sample["sim_time"] - lap_start), 6)
            sample["completion_in_lap"] = round(float(sample["absolute_completion"] - ((lap_index - 1) * 100.0)), 6)
            sample["lap_number"] = lap_index
            lap_samples.append(sample)
            cursor += 1

        if not lap_samples:
            reports.append(
                {
                    "lap": lap_index,
                    "lap_time": round(float(lap_time), 3),
                    "start_time": round(float(lap_start), 3),
                    "end_time": round(float(lap_end), 3),
                    "sample_count": 0,
                    "max_actual_speed": None,
                    "mean_actual_speed": None,
                    "max_abs_steer": None,
                    "max_distance_from_track": None,
                    "min_completion_in_lap": None,
                    "max_completion_in_lap": None,
                    "samples": [],
                }
            )
            elapsed_so_far = lap_end
            continue

        actual_speeds = [item["actual_speed"] for item in lap_samples]
        distances = [item["distance_from_track"] for item in lap_samples]
        steers = [abs(item["command_steer"]) for item in lap_samples]
        completions = [item["completion_in_lap"] for item in lap_samples]
        reports.append(
            {
                "lap": lap_index,
                "lap_time": round(float(lap_time), 3),
                "start_time": round(float(lap_start), 3),
                "end_time": round(float(lap_end), 3),
                "sample_count": len(lap_samples),
                "max_actual_speed": round(float(max(actual_speeds)), 3),
                "mean_actual_speed": round(float(sum(actual_speeds) / len(actual_speeds)), 3),
                "max_abs_steer": round(float(max(steers)), 3),
                "max_distance_from_track": round(float(max(distances)), 6),
                "min_completion_in_lap": round(float(min(completions)), 3),
                "max_completion_in_lap": round(float(max(completions)), 3),
                "samples": lap_samples,
            }
        )
        elapsed_so_far = lap_end

    return reports


def humanize_driver_name(value: str) -> str:
    return value.replace("_", " ")


def format_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def render_summary_cards(summary: dict) -> str:
    lap_spread = None
    if summary["best_lap"] is not None and summary["worst_lap"] is not None:
        lap_spread = summary["worst_lap"] - summary["best_lap"]
    status_label = "Finished clean" if summary["finished"] else summary["status"].replace("_", " ")
    cards = [
        ("Status", status_label),
        ("Lap Target", f"{summary['laps_completed']} / {summary['lap_target']}"),
        ("Race Time", f"{summary['total_time']:.3f}s"),
        ("Best Lap", f"{summary['best_lap']:.3f}s" if summary["best_lap"] is not None else "-"),
        ("Median Lap", f"{summary['median_lap']:.3f}s" if summary["median_lap"] is not None else "-"),
        ("Lap Spread", f"{lap_spread:.3f}s" if lap_spread is not None else "-"),
        ("Telemetry", f"{summary['sample_count']} samples"),
    ]
    return "".join(
        "<article class='metric-card'>"
        f"<div class='metric-label'>{escape(label)}</div>"
        f"<div class='metric-value'>{escape(value)}</div>"
        "</article>"
        for label, value in cards
    )


def render_lap_rows(lap_reports: list[dict]) -> str:
    rows = []
    for lap in lap_reports:
        row = (
            f"<tr data-lap-row='{lap['lap']}'>"
            f"<td><button class='lap-button' type='button' data-lap='{lap['lap']}'>Lap {lap['lap']}</button></td>"
            f"<td>{lap['lap_time']:.3f}s</td>"
        )
        if lap["mean_actual_speed"] is None:
            row += "<td>-</td><td>-</td><td>-</td><td>-</td>"
        else:
            row += (
                f"<td>{lap['mean_actual_speed']:.3f} m/s</td>"
                f"<td>{lap['max_actual_speed']:.3f} m/s</td>"
                f"<td>{lap['max_abs_steer']:.3f} rad</td>"
                f"<td>{lap['max_distance_from_track']:.4f}</td>"
            )
        row += "</tr>"
        rows.append(row)
    return "".join(rows)


def build_html_report(report: dict, destination: Path):
    summary = report["summary"]
    lap_reports = report["lap_reports"]
    generated_display = format_timestamp(summary["generated_at"])
    driver_display = humanize_driver_name(summary["driver_name"])
    payload_json = json.dumps(report, separators=(",", ":"))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FT Grand Prix Endurance Showcase</title>
  <style>
    :root {{
      --paper: #efe4d1;
      --sand: #ddc9a7;
      --ink: #1d2a31;
      --muted: #56646c;
      --panel: rgba(255, 250, 243, 0.82);
      --line: rgba(29, 42, 49, 0.12);
      --teal: #0d7c86;
      --amber: #c0652b;
      --olive: #5e7a51;
      --rose: #a53f4d;
      --shadow: 0 24px 60px rgba(44, 29, 8, 0.16);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font: 16px/1.5 "Avenir Next", "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.55), transparent 34%),
        radial-gradient(circle at 84% 12%, rgba(13,124,134,0.16), transparent 24%),
        linear-gradient(135deg, #d7c1a0 0%, var(--paper) 38%, #e6dccf 100%);
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
      background-size: 20px 20px;
      opacity: 0.26;
    }}
    main {{
      position: relative;
      max-width: 1420px;
      margin: 0 auto;
      padding: 32px 22px 56px;
    }}
    .stack {{
      display: grid;
      gap: 22px;
    }}
    .hero, .panel {{
      background: var(--panel);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255,255,255,0.55);
      border-radius: 28px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .hero-copy {{
      padding: 24px 26px 22px;
      background:
        linear-gradient(140deg, rgba(13,124,134,0.18), transparent 55%),
        linear-gradient(180deg, rgba(255,255,255,0.55), rgba(255,255,255,0));
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--teal);
      font-weight: 800;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.02;
      letter-spacing: -0.04em;
      text-transform: capitalize;
      max-width: 18ch;
    }}
    .lede {{
      margin: 0;
      max-width: 64ch;
      color: var(--muted);
      font-size: 15px;
    }}
    .footnote {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
    }}
    .footnote span {{
      min-width: 0;
    }}
    .footnote strong {{
      color: var(--ink);
      font-weight: 700;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
    }}
    .metric-card {{
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(255,255,255,0.55);
      border-radius: 22px;
      padding: 18px;
      min-height: 112px;
      box-shadow: 0 10px 24px rgba(44, 29, 8, 0.08);
    }}
    .metric-label {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .metric-value {{
      font-size: 28px;
      line-height: 1.05;
      font-weight: 800;
      letter-spacing: -0.04em;
      word-break: break-word;
    }}
    .panel-body {{
      padding: 0 22px 22px;
    }}
    .panel-header {{
      padding: 18px 22px 0;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }}
    .panel-title {{
      margin: 0;
      font-size: 20px;
      letter-spacing: -0.03em;
    }}
    .panel-kicker {{
      color: var(--muted);
      font-size: 13px;
      max-width: 38ch;
      text-align: right;
    }}
    .chart-wrap {{
      padding: 12px 18px 22px;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 290px;
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.86), rgba(252,246,239,0.92)),
        linear-gradient(135deg, rgba(13,124,134,0.06), rgba(192,101,43,0.08));
      border: 1px solid rgba(29, 42, 49, 0.08);
    }}
    .lap-bar-chart {{
      cursor: pointer;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: 0.78fr 1.22fr;
      gap: 22px;
      align-items: start;
    }}
    .table-shell {{
      padding: 10px 18px 22px;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }}
    th {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      font-weight: 700;
    }}
    tbody tr {{
      transition: background 120ms ease, color 120ms ease;
    }}
    tbody tr[data-lap-row].active {{
      background: rgba(13,124,134,0.08);
    }}
    .lap-button {{
      border: 0;
      background: transparent;
      color: var(--teal);
      padding: 0;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .lap-button.active {{
      color: var(--amber);
    }}
    .lap-toolbar {{
      padding: 0 22px 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }}
    .lap-toolbar select {{
      appearance: none;
      border: 1px solid rgba(29, 42, 49, 0.14);
      border-radius: 999px;
      background: rgba(255,255,255,0.9);
      padding: 10px 14px;
      min-width: 160px;
      font: inherit;
      color: var(--ink);
    }}
    .lap-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
      padding: 0 22px 18px;
    }}
    .lap-stat {{
      padding: 14px 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(29, 42, 49, 0.08);
    }}
    .lap-stat-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .lap-stat-value {{
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .lap-charts {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      padding: 0 22px 22px;
    }}
    .mini-chart h3 {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .mini-chart canvas {{
      height: 240px;
    }}
    .footer {{
      padding: 22px;
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.94em;
      white-space: normal;
      word-break: break-word;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 1100px) {{
      .detail-grid, .lap-charts {{
        grid-template-columns: 1fr;
      }}
      .panel-kicker {{
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="stack">
      <section class="hero hero-copy">
        <div class="eyebrow">Formula Trinity Endurance Report</div>
        <h1>{escape(driver_display)} on {escape(summary["track"])}</h1>
        <p class="lede">This page summarizes one clean verification run of the current submission candidate. Use the lap bars or the lap ledger to choose a lap, then inspect the corresponding speed, steering, distance-from-centerline, and XY path traces below.</p>
        <div class="footnote">
          <span><strong>Generated</strong> {escape(generated_display)}</span>
          <span><strong>Run folder</strong> <code>{escape(summary["run_folder_name"])}</code></span>
          <span><strong>Physics FPS</strong> {summary["physics_fps"]}</span>
          <span><strong>Sampling</strong> every {summary["sample_stride_steps"]} steps</span>
        </div>
      </section>

      <section class="metrics">
        {render_summary_cards(summary)}
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2 class="panel-title">Lap Pace</h2>
          <div class="panel-kicker">Each bar is one lap. Click any bar to jump the deep dive and ledger to that lap.</div>
        </div>
        <div class="chart-wrap"><canvas id="lapTimesChart" class="lap-bar-chart"></canvas></div>
      </section>

      <section class="panel detail-grid">
      <section>
        <div class="panel-header">
          <h2 class="panel-title">Lap Ledger</h2>
          <div class="panel-kicker">A compact table for comparing laps before drilling into one of them.</div>
        </div>
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Lap</th>
                <th>Lap Time</th>
                <th>Mean Speed</th>
                <th>Peak Speed</th>
                <th>Peak Steer</th>
                <th>Max Track Dist</th>
              </tr>
            </thead>
            <tbody>
              {render_lap_rows(lap_reports)}
            </tbody>
          </table>
        </div>
      </section>
      <section>
        <div class="panel-header">
          <h2 class="panel-title">Lap Deep Dive</h2>
          <div class="panel-kicker" id="lapSubtitle">Selected lap telemetry</div>
        </div>
        <div class="lap-toolbar">
          <label for="lapSelect">Lap</label>
          <select id="lapSelect"></select>
        </div>
        <div class="lap-stats" id="lapStats"></div>
        <div class="lap-charts">
          <section class="mini-chart">
            <h3>Speed</h3>
            <canvas id="speedChart"></canvas>
          </section>
          <section class="mini-chart">
            <h3>Steering</h3>
            <canvas id="steerChart"></canvas>
          </section>
          <section class="mini-chart">
            <h3>Distance From Centerline</h3>
            <canvas id="distanceChart"></canvas>
          </section>
          <section class="mini-chart">
            <h3>XY Path</h3>
            <canvas id="pathChart"></canvas>
          </section>
        </div>
      </section>
      </section>

      <section class="panel footer">
        Artifacts in this folder: <code>summary.json</code>, <code>telemetry.json</code>, <code>lap_times.csv</code>, and this <code>report.html</code>. Abort rules were: no progress for more than {summary["no_progress_sim_seconds"]} simulated seconds, or any single lap exceeding {summary["max_lap_seconds"]} simulated seconds.
      </section>
    </section>
  </main>

  <script>
    const report = {payload_json};
    const summary = report.summary;
    const lapReports = report.lap_reports;
    let currentLap = summary.best_lap_index || (lapReports[0] ? lapReports[0].lap : 1);
    let lapBarHitboxes = [];

    function formatSeconds(value) {{
      if (value == null || Number.isNaN(value)) return "-";
      return `${{value.toFixed(3)}}s`;
    }}

    function formatValue(value, digits = 3) {{
      if (value == null || Number.isNaN(value)) return "-";
      return value.toFixed(digits);
    }}

    function withCanvasSize(canvas) {{
      const ratio = Math.max(window.devicePixelRatio || 1, 1);
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.floor(rect.width));
      const height = Math.max(220, Math.floor(rect.height));
      if (canvas.width !== width * ratio || canvas.height !== height * ratio) {{
        canvas.width = width * ratio;
        canvas.height = height * ratio;
      }}
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {{ ctx, width, height }};
    }}

    function drawAxes(ctx, width, height, labels) {{
      const left = 56;
      const top = 18;
      const innerWidth = width - left - 18;
      const innerHeight = height - top - 32;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(255,255,255,0.55)";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(29,42,49,0.08)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i += 1) {{
        const y = top + (innerHeight * i) / 4;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(left + innerWidth, y);
        ctx.stroke();
      }}
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, top + innerHeight);
      ctx.lineTo(left + innerWidth, top + innerHeight);
      ctx.strokeStyle = "rgba(29,42,49,0.2)";
      ctx.stroke();

      ctx.fillStyle = "rgba(86,100,108,0.92)";
      ctx.font = "12px Avenir Next, IBM Plex Sans, sans-serif";
      ctx.fillText(labels.yMax, 10, top + 4);
      ctx.fillText(labels.yMin, 10, top + innerHeight + 4);
      ctx.fillText(labels.xMin, left, height - 10);
      const xMaxWidth = ctx.measureText(labels.xMax).width;
      ctx.fillText(labels.xMax, left + innerWidth - xMaxWidth, height - 10);
      return {{ left, top, innerWidth, innerHeight }};
    }}

    function drawLineChart(canvas, series, config = {{}}) {{
      const {{ ctx, width, height }} = withCanvasSize(canvas);
      const xValues = series.flatMap((item) => item.points.map((point) => point.x));
      const yValues = series.flatMap((item) => item.points.map((point) => point.y));
      if (!xValues.length || !yValues.length) {{
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#56646c";
        ctx.font = "14px Avenir Next, IBM Plex Sans, sans-serif";
        ctx.fillText("No data", 18, 24);
        return;
      }}
      const minX = Math.min(...xValues);
      const maxX = Math.max(...xValues);
      let minY = Math.min(...yValues);
      let maxY = Math.max(...yValues);
      if (config.minY != null) minY = Math.min(minY, config.minY);
      if (config.maxY != null) maxY = Math.max(maxY, config.maxY);
      if (Math.abs(maxY - minY) < 1e-9) {{
        maxY += 1;
        minY -= 1;
      }}
      const chart = drawAxes(ctx, width, height, {{
        yMax: `${{maxY.toFixed(2)}}${{config.ySuffix || ""}}`,
        yMin: `${{minY.toFixed(2)}}${{config.ySuffix || ""}}`,
        xMin: `${{minX.toFixed(1)}}${{config.xSuffix || ""}}`,
        xMax: `${{maxX.toFixed(1)}}${{config.xSuffix || ""}}`,
      }});
      const xSpread = Math.max(maxX - minX, 1e-9);
      const ySpread = Math.max(maxY - minY, 1e-9);
      series.forEach((item) => {{
        if (!item.points.length) return;
        ctx.beginPath();
        item.points.forEach((point, index) => {{
          const x = chart.left + ((point.x - minX) / xSpread) * chart.innerWidth;
          const y = chart.top + chart.innerHeight - ((point.y - minY) / ySpread) * chart.innerHeight;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.lineWidth = item.lineWidth || 2.4;
        ctx.strokeStyle = item.color;
        ctx.stroke();
      }});
      if (config.legend) {{
        ctx.font = "12px Avenir Next, IBM Plex Sans, sans-serif";
        let x = chart.left;
        config.legend.forEach((entry) => {{
          ctx.fillStyle = entry.color;
          ctx.fillRect(x, 10, 16, 3);
          ctx.fillStyle = "#56646c";
          ctx.fillText(entry.label, x + 22, 16);
          x += 22 + ctx.measureText(entry.label).width + 24;
        }});
      }}
    }}

    function drawLapBarChart(canvas, selectedLap) {{
      const lapTimes = summary.lap_times || [];
      const {{ ctx, width, height }} = withCanvasSize(canvas);
      if (!lapTimes.length) {{
        ctx.clearRect(0, 0, width, height);
        return;
      }}
      const minLap = Math.min(...lapTimes);
      const maxLap = Math.max(...lapTimes);
      const meanLap = lapTimes.reduce((total, value) => total + value, 0) / lapTimes.length;
      const lapSpread = Math.max(maxLap - minLap, 0.25);
      const yPadding = Math.max(0.18, lapSpread * 0.18);
      const yMin = Math.max(0, minLap - yPadding);
      const yMax = maxLap + yPadding * 1.15;
      const chart = drawAxes(ctx, width, height, {{
        yMax: `${{yMax.toFixed(2)}}s`,
        yMin: `${{yMin.toFixed(2)}}s`,
        xMin: "",
        xMax: "",
      }});
      const ySpread = Math.max(yMax - yMin, 1e-9);
      const gap = Math.max(4, Math.min(10, chart.innerWidth / (lapTimes.length * 4)));
      const barWidth = Math.max(10, (chart.innerWidth - gap * (lapTimes.length - 1)) / lapTimes.length);
      lapBarHitboxes = [];

      const meanY = chart.top + chart.innerHeight - ((meanLap - yMin) / ySpread) * chart.innerHeight;
      ctx.strokeStyle = "rgba(13,124,134,0.45)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(chart.left, meanY);
      ctx.lineTo(chart.left + chart.innerWidth, meanY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(13,124,134,0.9)";
      ctx.font = "12px Avenir Next, IBM Plex Sans, sans-serif";
      ctx.fillText(`Mean ${{meanLap.toFixed(3)}}s`, chart.left + 8, meanY - 8);

      lapTimes.forEach((lapTime, index) => {{
        const lap = index + 1;
        const x = chart.left + index * (barWidth + gap);
        const y = chart.top + chart.innerHeight - ((lapTime - yMin) / ySpread) * chart.innerHeight;
        const barHeight = chart.top + chart.innerHeight - y;
        const isSelected = lap === selectedLap;
        const isBest = lap === summary.best_lap_index;
        ctx.fillStyle = isSelected ? "#c0652b" : isBest ? "#0d7c86" : "rgba(29,42,49,0.30)";
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, Math.max(3, barHeight), 8);
        ctx.fill();
        if (lapTimes.length <= 30 && (lap === 1 || lap === lapTimes.length || lap % 5 === 0 || isSelected)) {{
          ctx.fillStyle = "rgba(86,100,108,0.92)";
          const label = String(lap);
          const labelWidth = ctx.measureText(label).width;
          const desiredX = x + barWidth / 2 - labelWidth / 2;
          const labelX = Math.min(chart.left + chart.innerWidth - labelWidth, Math.max(chart.left, desiredX));
          ctx.fillText(label, labelX, height - 10);
        }}
        lapBarHitboxes.push({{ lap, x, y, width: barWidth, height: Math.max(3, barHeight) }});
      }});
    }}

    function drawPathChart(canvas, lap) {{
      const {{ ctx, width, height }} = withCanvasSize(canvas);
      const points = lap.samples.map((item) => [item.pos_x, item.pos_y]);
      if (!points.length) {{
        ctx.clearRect(0, 0, width, height);
        return;
      }}
      const xs = points.map((point) => point[0]);
      const ys = points.map((point) => point[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const spanX = Math.max(maxX - minX, 1e-6);
      const spanY = Math.max(maxY - minY, 1e-6);
      const scale = Math.min((width - 36) / spanX, (height - 36) / spanY);
      const offsetX = (width - spanX * scale) / 2;
      const offsetY = (height - spanY * scale) / 2;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(255,255,255,0.62)";
      ctx.fillRect(0, 0, width, height);
      ctx.beginPath();
      points.forEach((point, index) => {{
        const x = offsetX + (point[0] - minX) * scale;
        const y = height - offsetY - (point[1] - minY) * scale;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.lineWidth = 2.4;
      ctx.strokeStyle = "#0d7c86";
      ctx.stroke();
      const start = points[0];
      const end = points[points.length - 1];
      const startX = offsetX + (start[0] - minX) * scale;
      const startY = height - offsetY - (start[1] - minY) * scale;
      const endX = offsetX + (end[0] - minX) * scale;
      const endY = height - offsetY - (end[1] - minY) * scale;
      ctx.fillStyle = "#c0652b";
      ctx.beginPath();
      ctx.arc(startX, startY, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#a53f4d";
      ctx.beginPath();
      ctx.arc(endX, endY, 5, 0, Math.PI * 2);
      ctx.fill();
    }}

    function renderOverallCharts() {{
      drawLapBarChart(document.getElementById("lapTimesChart"), currentLap);
    }}

    function setActiveLapButton(lapNumber) {{
      document.querySelectorAll(".lap-button").forEach((button) => {{
        button.classList.toggle("active", Number(button.dataset.lap) === lapNumber);
      }});
      document.querySelectorAll("tr[data-lap-row]").forEach((row) => {{
        row.classList.toggle("active", Number(row.dataset.lapRow) === lapNumber);
      }});
    }}

    function renderLapStats(lap) {{
      const stats = [
        ["Lap Time", formatSeconds(lap.lap_time)],
        ["Mean Speed", `${{formatValue(lap.mean_actual_speed)}} m/s`],
        ["Peak Speed", `${{formatValue(lap.max_actual_speed)}} m/s`],
        ["Peak Steer", `${{formatValue(lap.max_abs_steer)}} rad`],
        ["Max Track Dist", formatValue(lap.max_distance_from_track, 4)],
        ["Samples", `${{lap.sample_count}}`],
      ];
      document.getElementById("lapStats").innerHTML = stats.map(([label, value]) => `
        <article class="lap-stat">
          <div class="lap-stat-label">${{label}}</div>
          <div class="lap-stat-value">${{value}}</div>
        </article>
      `).join("");
    }}

    function renderLapCharts(lap) {{
      drawLineChart(document.getElementById("speedChart"), [
        {{
          color: "#0d7c86",
          points: lap.samples.map((sample) => ({{ x: sample.elapsed, y: sample.actual_speed }})),
        }},
        {{
          color: "#c0652b",
          points: lap.samples.map((sample) => ({{ x: sample.elapsed, y: sample.command_speed }})),
          lineWidth: 2.0,
        }},
      ], {{
        xSuffix: "s",
        ySuffix: "m/s",
        legend: [
          {{ label: "Actual", color: "#0d7c86" }},
          {{ label: "Command", color: "#c0652b" }},
        ],
      }});
      drawLineChart(document.getElementById("steerChart"), [
        {{
          color: "#a53f4d",
          points: lap.samples.map((sample) => ({{ x: sample.elapsed, y: sample.command_steer }})),
        }},
      ], {{
        xSuffix: "s",
        ySuffix: "rad",
      }});
      drawLineChart(document.getElementById("distanceChart"), [
        {{
          color: "#5e7a51",
          points: lap.samples.map((sample) => ({{ x: sample.elapsed, y: sample.distance_from_track }})),
        }},
        {{
          color: "rgba(29,42,49,0.4)",
          points: [
            {{ x: 0, y: 1.0 }},
            {{ x: lap.lap_time, y: 1.0 }},
          ],
          lineWidth: 1.5,
        }},
      ], {{
        xSuffix: "s",
        ySuffix: "",
        legend: [
          {{ label: "Distance", color: "#5e7a51" }},
          {{ label: "Off-track threshold", color: "rgba(29,42,49,0.4)" }},
        ],
      }});
      drawPathChart(document.getElementById("pathChart"), lap);
    }}

    function renderLap(lapNumber) {{
      const lap = lapReports.find((item) => item.lap === lapNumber) || lapReports[0];
      if (!lap) return;
      currentLap = lap.lap;
      document.getElementById("lapSubtitle").textContent =
        `Lap ${{lap.lap}} ran from ${{lap.start_time.toFixed(3)}}s to ${{lap.end_time.toFixed(3)}}s of the race clock.`;
      document.getElementById("lapSelect").value = String(lap.lap);
      setActiveLapButton(lap.lap);
      renderOverallCharts();
      renderLapStats(lap);
      renderLapCharts(lap);
    }}

    function initLapControls() {{
      const select = document.getElementById("lapSelect");
      select.innerHTML = lapReports.map((lap) => `<option value="${{lap.lap}}">Lap ${{lap.lap}}</option>`).join("");
      select.addEventListener("change", (event) => renderLap(Number(event.target.value)));
      document.querySelectorAll(".lap-button").forEach((button) => {{
        button.addEventListener("click", () => renderLap(Number(button.dataset.lap)));
      }});
      const lapChart = document.getElementById("lapTimesChart");
      lapChart.addEventListener("click", (event) => {{
        const rect = lapChart.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const hit = lapBarHitboxes.find((item) => x >= item.x && x <= item.x + item.width && y >= item.y && y <= item.y + item.height);
        if (hit) renderLap(hit.lap);
      }});
      renderLap(currentLap);
    }}

    window.addEventListener("load", () => {{
      initLapControls();
    }});
    window.addEventListener("resize", () => {{
      renderLap(currentLap);
    }});
  </script>
</body>
</html>
"""
    destination.write_text(html, encoding="utf-8")


def run_showcase(driver_path: str, lap_target: int, track: str, physics_fps: int, wall_timeout: float, no_progress_sim_seconds: float, max_lap_seconds: float, sample_stride_steps: int) -> dict:
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
    timestep = float(mj.model.opt.timestep)
    best_absolute_completion = float(vehicle_state.absolute_completion())
    last_progress_steps = 0
    start_wall = time.time()
    status = "timeout"
    stall_reason = None
    samples = []
    lap_events = []

    initial_snapshot = vehicle_state.snapshot(time=mj.steps / timestep)
    samples.append(make_sample(mj.steps, timestep, vehicle_state, float(initial_snapshot.yaw)))

    while time.time() < start_wall + wall_timeout:
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
                    completed_lap = vehicle_state.laps + 1
                    if vehicle_state.good_start:
                        vehicle_state.times.append(lap_time)
                        vehicle_state.start = mj.steps
                        lap_events.append(
                            {
                                "lap": int(completed_lap),
                                "lap_time": round(float(lap_time), 3),
                                "sim_time": round(float(mj.steps * timestep), 3),
                            }
                        )
                    vehicle_state.laps += 1
                    vehicle_state.good_start = True
            vehicle_state.completion = completion
            absolute_completion = float(vehicle_state.absolute_completion())
            if absolute_completion > best_absolute_completion:
                best_absolute_completion = absolute_completion
                last_progress_steps = mj.steps
            if vehicle_state.laps >= mj.option("lap_target"):
                vehicle_state.finished = True
                status = "finished"
                break

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

        if mj.steps % sample_stride_steps == 0:
            post_snapshot = vehicle_state.snapshot(time=mj.steps / timestep)
            samples.append(make_sample(mj.steps, timestep, vehicle_state, float(post_snapshot.yaw)))

        no_progress_sim = (mj.steps - last_progress_steps) * timestep
        if no_progress_sim > no_progress_sim_seconds:
            status = "stalled"
            stall_reason = f"no progress for {no_progress_sim:.1f}s sim"
            break

        current_lap_time = (mj.steps - vehicle_state.start) * timestep
        if current_lap_time > max_lap_seconds:
            status = "lap_timeout"
            stall_reason = f"current lap exceeded {max_lap_seconds:.1f}s sim"
            break

    final_snapshot = vehicle_state.snapshot(time=mj.steps / timestep)
    samples.append(make_sample(mj.steps, timestep, vehicle_state, float(final_snapshot.yaw)))

    lap_times = [round(float(value), 3) for value in vehicle_state.times]
    best_lap = round(float(min(vehicle_state.times)), 3) if vehicle_state.times else None
    median_lap = round(float(median(vehicle_state.times)), 3) if vehicle_state.times else None
    worst_lap = round(float(max(vehicle_state.times)), 3) if vehicle_state.times else None
    best_lap_index = None
    if best_lap is not None:
        best_lap_index = lap_times.index(best_lap) + 1

    return {
        "summary": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "driver": driver_path,
            "driver_name": driver_path.split(".")[-1],
            "track": track,
            "lap_target": int(lap_target),
            "physics_fps": int(physics_fps),
            "sample_stride_steps": int(sample_stride_steps),
            "status": status,
            "finished": bool(vehicle_state.finished),
            "laps_completed": int(vehicle_state.laps),
            "lap_times": lap_times,
            "total_time": round(float(sum(vehicle_state.times)), 3),
            "best_lap": best_lap,
            "median_lap": median_lap,
            "worst_lap": worst_lap,
            "best_lap_index": best_lap_index,
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
            "wall_time": round(time.time() - start_wall, 3),
            "stall_reason": stall_reason,
            "sample_count": len(samples),
            "no_progress_sim_seconds": float(no_progress_sim_seconds),
            "max_lap_seconds": float(max_lap_seconds),
        },
        "samples": samples,
        "lap_events": lap_events,
    }


def write_outputs(report: dict, output_dir: Path):
    summary = dict(report["summary"])
    summary["run_folder_name"] = output_dir.name
    report["summary"] = summary
    report["lap_reports"] = compute_lap_reports(report["samples"], summary["lap_times"])

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "telemetry.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "lap_times.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lap", "lap_time_seconds"])
        for index, value in enumerate(summary["lap_times"], start=1):
            writer.writerow([index, value])
    build_html_report(report, output_dir / "report.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="drivers.daboss_endurance_driver")
    parser.add_argument("--laps", type=int, default=25)
    parser.add_argument("--track", default="track")
    parser.add_argument("--physics-fps", type=int, default=500)
    parser.add_argument("--wall-timeout", type=float, default=2400.0)
    parser.add_argument("--no-progress-sim-seconds", type=float, default=60.0)
    parser.add_argument("--max-lap-seconds", type=float, default=60.0)
    parser.add_argument("--sample-stride-steps", type=int, default=50)
    parser.add_argument("--output-root", default="race_runs")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = REPO_ROOT / args.output_root / f"showcase-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_showcase(
        driver_path=args.driver,
        lap_target=args.laps,
        track=args.track,
        physics_fps=args.physics_fps,
        wall_timeout=args.wall_timeout,
        no_progress_sim_seconds=args.no_progress_sim_seconds,
        max_lap_seconds=args.max_lap_seconds,
        sample_stride_steps=args.sample_stride_steps,
    )
    write_outputs(report, output_dir)
    print(
        json.dumps(
            {
                "event": "showcase_ready",
                "output_dir": str(output_dir),
                "report_html": str(output_dir / "report.html"),
                "summary_json": str(output_dir / "summary.json"),
                "telemetry_json": str(output_dir / "telemetry.json"),
                "lap_times_csv": str(output_dir / "lap_times.csv"),
                "status": report["summary"]["status"],
                "laps_completed": report["summary"]["laps_completed"],
                "total_time": report["summary"]["total_time"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
