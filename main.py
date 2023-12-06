import os
import pandas as pd

from collector import Collector
from processor import Processor
from regressor import Regressor
from config import DATASETS_DIRECTORY, DATASET_FILENAME, MODELS, OUTPUT_DIRECTORY


def main():
    filepath = os.path.join(os.getcwd(), DATASETS_DIRECTORY, DATASET_FILENAME)
    raw_data = pd.read_csv(filepath, delimiter=",")

    processor = Processor(raw_data)
    processor.do_process()

    X_train = processor.X_train
    y_train = processor.y_train
    data = processor.data

    for definition in MODELS:
        model_class = list(definition.keys())[0]
        model_obj = model_class()
        model_params = definition[model_class]

        collector = Collector(model_obj.__class__.__name__)

        model = Regressor(data, model_obj, model_params, X_train, y_train)
        model.train(collector)

        collector.wrive_csv()
        collector.draw_line_chart()
        collector.draw_cumulative_pnl_chart()


if __name__ == "__main__":
    main()
