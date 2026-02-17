#!/usr/bin/env python3
"""
Parse validation results from JSON output files.

Reads structured JSON results from JobRecorder instead of parsing log text.
"""
import argparse
import glob
import json
import sys
from pathlib import Path


def find_json_results(log_dir="."):
    """Find validation_results_*.json files in directory"""
    pattern = str(Path(log_dir) / "results/validation_results_*.json")
    json_files = glob.glob(pattern)
    return sorted(json_files, reverse=True)  # Most recent first


def load_results_from_json(json_file):
    """Load validation results from JSON file"""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Parse validation results from JSON")
    parser.add_argument("--log-file", required=True, help="Path to validation log file (for backward compat)")
    parser.add_argument("--runs-json", required=True, help="JSON string of expected runs")
    parser.add_argument("--platform", required=True, help="Platform name (K8, SLURM, SMJOBS)")
    parser.add_argument("--output-file", help="Optional output file for markdown summary")
    parser.add_argument("--json-dir", default=".", help="Directory to search for JSON results")
    parser.add_argument("--custom-flag", default="NA", help="Custom user flag to identify the run")

    args = parser.parse_args()

    # Find and load JSON results
    json_files = find_json_results(args.json_dir)

    if not json_files:
        print(f"❌ No validation JSON results found in {args.json_dir}", file=sys.stderr)
        print(f"Expected files matching: validation_results_*.json", file=sys.stderr)
        sys.exit(1)

    all_jobs = []
    has_failures = False

    # Load from JSON files
    for json_file in json_files:
        results = load_results_from_json(json_file)
        if results and "jobs" in results:
            all_jobs.extend(results["jobs"])

            # Check for failures
            metadata = results.get("metadata", {})
            if metadata.get("failed_jobs", 0) > 0:
                has_failures = True

    # Generate markdown summary from JSON data
    markdown_lines = []
    markdown_lines.append("")
    markdown_lines.append(f"## 📊 Validation Results - {args.platform}")
    markdown_lines.append("")
    custom_flag = args.custom_flag if args.custom_flag else "NA"

    markdown_lines.append(
        "| # | Recipe | Instance | Status | Duration | Tokens | Throughput | Custom Flag | Output Log |"
    )
    markdown_lines.append(
        "|---|--------|----------|--------|----------|--------|------------|-------------|------------|"
    )

    if not all_jobs:
        markdown_lines.append("| - | - | - | ⚠️ No jobs found in JSON | - | - | - | - | - |")
    else:
        for idx, job in enumerate(all_jobs, 1):
            filename = Path(job.get("input_filename", "Unknown")).name
            instance = job.get("instance_type", "N/A")
            status = job.get("status", "Unknown")
            throughput = job.get("tokens_throughput", "N/A")
            throughput_data = job.get("throughput_data", {})
            duration = throughput_data.get("duration", "N/A") if throughput_data else "N/A"
            tokens = throughput_data.get("tokens", "N/A") if throughput_data else "N/A"
            output_log = job.get("output_log", "N/A")

            # Format throughput
            if throughput and throughput != "N/A":
                throughput_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else str(throughput)
            else:
                throughput_str = "N/A"

            # Format output_log for markdown (escape pipes, truncate if too long)
            if output_log and output_log != "N/A":
                output_log_str = str(output_log).replace("|", "\\|")
                if len(output_log_str) > 100:
                    output_log_str = output_log_str[:97] + "..."
            else:
                output_log_str = "N/A"

            # Determine emoji
            if status.lower() == "complete":
                status_display = "✅ Complete"
            elif status.lower() == "failed":
                status_display = "❌ Failed"
                has_failures = True
            else:
                status_display = f"⚠️ {status}"

            markdown_lines.append(
                f"| {idx} | {filename} | {instance} | {status_display} | {duration} | {tokens} | {throughput_str} | {custom_flag} | {output_log_str} |"
            )

    markdown_lines.append("")

    # Output
    output_text = "\n".join(markdown_lines)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"✅ Wrote summary to {args.output_file}")
    else:
        print(output_text)

    # Exit with non-zero status if any tests failed
    if has_failures:
        print(f"❌ One or more validation tests failed", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
