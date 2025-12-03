# A common utility class for the validation scripts to output the job results in a more readable manner
#  Sample results output :-
# +---+---------------+---------------+----------+-----------------------+
# | # | InputFileName | OutputPath    | Status   | OutputLog             |
# +---+---------------+---------------+----------+-----------------------+
# | 1 | data1.txt     | /results/job1 | Complete | Success               |
# | 2 | data2.txt     | /results/job2 | Failed   | Error: File not found |
# +---+---------------+---------------+----------+-----------------------+
# Relevant functions:- add_job(), update_job(), print_results()

import csv
import textwrap
from datetime import datetime


class Job:
    def __init__(self, input_filename="", output_path="", status="", output_log="", tokens_throughput=None):
        self.input_filename = input_filename
        self.output_path = output_path
        self.status = status
        self.output_log = output_log
        self.tokens_throughput = tokens_throughput
        # Additional throughput metadata
        self.throughput_data = {}
        self.job_success_status = None
        self.platform = None
        self.instance_type = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class JobRecorder:
    def __init__(self, max_col_width=30):
        self.jobs = {}
        self.max_col_width = max_col_width

    # Use the input filename as the key for each job
    def add_job(self, input_filename, output_path="", status="", output_log="", tokens_throughput=None):
        job = Job(input_filename, output_path, status, output_log, tokens_throughput)
        self.jobs[input_filename] = job
        return input_filename

    def update_job(self, input_filename, **kwargs):
        if input_filename in self.jobs:
            self.jobs[input_filename].update(**kwargs)

    # Just to prevent the table from becoming too wide
    def _wrap_text(self, text, width):
        return textwrap.fill(text, width=width).split("\n")

    def get_status(self):
        return [[self.jobs[job].input_filename, self.jobs[job].status] for job in self.jobs]

    def print_results(self):
        if not self.jobs:
            print("No jobs recorded")
            return

        headers = ["#", "InputFileName", "OutputPath", "Status", "Tokens/Sec", "OutputLog"]

        # Prepare wrapped data
        wrapped_data = []
        for i, job in enumerate(self.jobs.values()):
            # Format tokens throughput
            tokens_display = str(job.tokens_throughput) if job.tokens_throughput is not None else "N/A"

            row_data = [
                [str(i + 1)],
                self._wrap_text(job.input_filename, self.max_col_width),
                self._wrap_text(job.output_path, self.max_col_width),
                self._wrap_text(job.status, self.max_col_width),
                self._wrap_text(tokens_display, self.max_col_width),
                self._wrap_text(job.output_log, self.max_col_width),
            ]
            wrapped_data.append(row_data)

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row_data in wrapped_data:
            for col_idx, col_lines in enumerate(row_data):
                max_line_width = max(len(line) for line in col_lines) if col_lines else 0
                widths[col_idx] = max(widths[col_idx], max_line_width)

        # Print table
        separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths)) + "|"

        print(separator)
        print(header_row)
        print(separator)

        for row_idx, row_data in enumerate(wrapped_data):
            max_lines = max(len(col_lines) for col_lines in row_data)

            for line_idx in range(max_lines):
                row_parts = []
                for col_idx, col_lines in enumerate(row_data):
                    text = col_lines[line_idx] if line_idx < len(col_lines) else ""
                    row_parts.append(f" {text:<{widths[col_idx]}} ")
                print("|" + "|".join(row_parts) + "|")

            # Add separator after each row (except the last one)
            if row_idx < len(wrapped_data) - 1:
                print(separator)

        print(separator)

    def write_throughput_csv(self, filename=None):
        """Write consolidated throughput data from all jobs to CSV file"""

        try:
            # Create CSV filename with timestamp if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"throughput_results_{timestamp}.csv"

            # Define CSV headers (all possible fields)
            headers = [
                "job_name",
                "job_status",
                "platform",
                "instance_type",
                "status",
                "model",
                "dataset",
                "duration",
                "tokens",
                "throughput",
                "batch_size",
                "timestamp",
            ]

            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                # Write data for each job
                for job in self.jobs.values():
                    # Get throughput data from job
                    throughput_data = getattr(job, "throughput_data", {})

                    # Prepare row data with enhanced fields
                    row_data = {
                        "job_name": job.input_filename,
                        "job_status": "Success" if getattr(job, "job_success_status", None) else "Failed",
                        "platform": getattr(job, "platform", "Unknown"),
                        "instance_type": getattr(job, "instance_type", ""),
                        "status": throughput_data.get("status", ""),
                        "model": throughput_data.get("model", ""),
                        "dataset": throughput_data.get("dataset", ""),
                        "duration": throughput_data.get("duration", ""),
                        "tokens": throughput_data.get("tokens", ""),
                        "throughput": throughput_data.get("throughput", ""),
                        "batch_size": throughput_data.get("batch_size", ""),
                        "timestamp": datetime.now().isoformat(),
                    }

                    writer.writerow(row_data)

                print(f"Consolidated throughput results written to {filename}")
                return filename

        except Exception as e:
            print(f"Error writing consolidated CSV file: {e}")
            return None
