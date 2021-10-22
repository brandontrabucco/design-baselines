import tensorflow as tf
import numpy as np
import argparse
import os
import random
import string
import pickle as pkl
import glob
import design_bench as db
from datetime import datetime


if __name__ == "__main__":

    baseline_to_iteration = {
        "autofocused-cbas": 20,
        "cbas": 20,
        "bo-qei": 10,
        "cma-es": 0,
        "gradient-ascent": 200,
        "gradient-ascent-min-ensemble": 200,
        "gradient-ascent-mean-ensemble": 200,
        "mins": 0,
        "reinforce": 200
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/home/btrabucco/mbo-results")
    parser.add_argument("--input-pkl", type=str, default="final_output.pkl")
    parser.add_argument("--out", type=str, default="/home/btrabucco/final-results")
    args = parser.parse_args()

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    digits_and_letters = list(string.digits) + list(string.ascii_letters)

    task = db.make("CIFARNAS-Exact-v0")
    task.map_to_logits()
    task.map_normalize_x()

    with open(args.input_pkl, "rb") as f:
        input_dict = pkl.load(f)

    for baseline, step in baseline_to_iteration.items():

        solution_files = glob.glob(os.path.join(
            args.input_dir, f"{baseline}-nas/*/*/data/solution.npy"))

        step = tf.cast(tf.convert_to_tensor(
            baseline_to_iteration[baseline]), tf.int64)

        for solution_file in solution_files:
            solution = np.load(solution_file)

            if solution.shape == (128, 64, 4):
                solution = task.to_integers(task.denormalize_x(solution))

            perf = [input_dict["-".join([str(xi) for xi in x])]
                    for x in solution.tolist()]

            perf_50 = np.percentile(perf, 50)
            perf_80 = np.percentile(perf, 80)
            perf_90 = np.percentile(perf, 90)
            perf_100 = np.percentile(perf, 100)

            print(baseline, perf_50, perf_80, perf_90, perf_100)

            sweep_folder_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(solution_file))))
            seed_folder_name = os.path.basename(os.path.dirname(os.path.dirname(solution_file)))

            output_dir = os.path.join(
                args.out,
                f"{baseline}-cifar-nas/"
                f"{sweep_folder_name}/{seed_folder_name}/data/")

            step = tf.cast(tf.convert_to_tensor(
                baseline_to_iteration[baseline]), tf.int64)

            tf.io.gfile.makedirs(output_dir)
            writer = tf.summary.create_file_writer(output_dir)

            with writer.as_default():

                tf.summary.scalar(
                    f'score/50th', perf_50, step=step)
                tf.summary.scalar(
                    f'score/80th', perf_80, step=step)
                tf.summary.scalar(
                    f'score/90th', perf_90, step=step)
                tf.summary.scalar(
                    f'score/100th', perf_100, step=step)
