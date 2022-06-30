import argparse
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument("--athena-data", type=str)
args = parser.parse_args()


dataset = pd.read_parquet(args.athena_data, engine="pyarrow")

train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(val_df, test_size=0.05, random_state=42)

# Write train, validation splits to output path
train_output_path = pathlib.Path("/opt/ml/processing/output/train")
val_output_path = pathlib.Path("/opt/ml/processing/output/validation")
test_output_path = pathlib.Path("/opt/ml/processing/output/test")
baseline_output_path = pathlib.Path("/opt/ml/processing/output/baseline")

train_df.to_csv(train_output_path / "train.csv", index=False)
val_df.to_csv(val_output_path / "validation.csv", index=False, header=False)
test_df.to_csv(test_output_path / "test.csv", index=False, header=False)

baseline = list(train_df.agg(
    {
        'account_length': 'mean', 
        'vmail_message': 'mean', 
        'day_mins': 'mean',
        'day_calls': 'mean', 
        'eve_mins': 'mean', 
        'eve_calls': 'mean', 
        'night_mins': 'mean', 
        'night_calls': 'mean',
        'intl_mins': 'mean', 
        'intl_calls': 'mean', 
        'custserv_calls': 'mean', 
        'pastsenti_nut': 'mean', 
        'pastsenti_pos': 'mean', 
        'pastsenti_neg': 'mean', 
        'mth_remain': 'mean',
        'int_l_plan_no': 'mean', 
        'int_l_plan_yes': 'mean', 
        'vmail_plan_no': 'mode', 
        'vmail_plan_yes': 'mode',
    }
).iloc[0])

pd.DataFrame(baseline).to_csv(baseline_output_path / 'baseline.csv', index=False, header=None)