import argparse

import pandas as pd
import numpy as np

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument("--athena-data", type=str)
args = parser.parse_args()


dataset = pd.read_parquet(args.athena_data, engine="pyarrow")


for idx, chunk in enumerate(np.array_split(dataset, 5)):
    chunk.to_csv(
        f'/opt/ml/processing/output/batch/batch_file_{idx}.csv',
        index=False, header=False
        )