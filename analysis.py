import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import glob
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=str)
    parser.add_argument("--hue", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    matching_files = glob.glob(os.path.join(args.local_dir, "eval_*.csv"))
    matching_files = [pd.read_csv(x) for x in matching_files]
    df = pd.concat(matching_files)

    sns.set(style='darkgrid')
    sns.lineplot(x='SGD Steps', y='Average Return', hue=args.hue, data=df, style='Type')
    plt.savefig(args.out)
