import os
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from config import OUTPUT_DIRECTORY, ROUND_DECIMALS


@dataclass
class Collector:
    model: str = "model"
    dates: list = field(default_factory=list)
    uuids: list = field(default_factory=list)
    maes: list = field(default_factory=list)
    best_params: list = field(default_factory=list)
    y_trues: list = field(default_factory=list)
    y_preds: list = field(default_factory=list)
    changes: list = field(default_factory=list)
    pnls: list = field(default_factory=list)
    seconds_taken: list = field(default_factory=list)

    @staticmethod
    def get_output_directory_path():
        return os.path.join(os.getcwd(), OUTPUT_DIRECTORY)

    def calculate_cumulative_pnls(self):
        return np.cumsum(self.pnls).round(ROUND_DECIMALS)

    def to_csv(self, filename, dataframe):
        dataframe.to_csv(filename, index=False)

    def format_best_params(self):
        return [
            ", ".join(f"{item[0]}: {item[1]}" for item in param.items())
            for param in self.best_params
        ]

    def calculate_mean(self):
        # Create a filtered version of the 'pnl' column with only numeric values
        cumulative_pnls = self.calculate_cumulative_pnls()
        numeric_pnl = pd.to_numeric(cumulative_pnls, errors="coerce")
        # Calculate the mean of the numeric values
        return numeric_pnl.mean()

    def calculate_stddev(self):
        cumulative_pnls = self.calculate_cumulative_pnls()
        numeric_pnl = pd.to_numeric(cumulative_pnls, errors="coerce")
        return numeric_pnl.std(ddof=0)

    def calculate_sharpe_ratio(self):
        mean_value = self.calculate_mean()
        stddev_value = self.calculate_stddev()
        return (mean_value / stddev_value) * math.sqrt(252)

    def wrive_csv(self):
        output_dir_path = Collector.get_output_directory_path()

        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        output_filepath = os.path.join(output_dir_path, f"{self.model}.csv")

        if os.path.exists(output_filepath) and os.path.isfile(output_filepath):
            os.remove(output_filepath)

        self.to_csv(
            output_filepath,
            pd.DataFrame(
                {
                    "ID": self.uuids,
                    "Date": self.dates,
                    "Target true": self.y_trues,
                    "Target predicted": self.y_preds,
                    "Change": self.changes,
                    "MAE": self.maes,
                    "PNL": self.pnls,
                    "Cumulative PNL": self.calculate_cumulative_pnls(),
                    "Seconds taken": self.seconds_taken,
                    "Best params": self.format_best_params(),
                }
            ),
        )

    def draw_cumulative_pnl_chart(self):
        output_dir_path = Collector.get_output_directory_path()

        output_filepath = os.path.join(output_dir_path, f"{self.model}.csv")
        df = pd.read_csv(output_filepath, delimiter=",")

        x = df["ID"]
        y = df["Cumulative PNL"]

        plt.figure(figsize=(35, 20))
        plt.plot(x, y, linewidth=2)

        font = {
            "family": "monospace",
            "weight": "normal",
            "size": 24,
        }
        plt.xlabel("ID", fontdict=font)
        plt.xticks(fontsize=16)
        plt.ylabel("PNL", fontdict=font)
        plt.yticks(fontsize=16)
        plt.title(f"{self.model} Cumulative PNL", fontdict=font)
        plt.grid()
        plt.savefig(os.path.join(output_dir_path, f"{self.model}_pnl.pdf"))
