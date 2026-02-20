import os, json


class Logger:
    """
    Handles metric logging.
    Keeps Trainer clean.
    """

    def __init__(self, run_dir):
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        os.makedirs(run_dir, exist_ok=True)

    def log(self, data: dict):
        """
        Logs metrics to console and file.
        """
        print(data)

        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(data) + "\n")
