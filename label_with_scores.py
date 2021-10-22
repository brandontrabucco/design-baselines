import pickle as pkl
import numpy as np
import os
import glob
import design_bench as db


if __name__ == "__main__":

    task = db.make("CIFARNAS-Exact-v0")
    task.map_to_logits()
    task.map_normalize_x()

    with open('final_output.pkl', 'rb') as f:
        final_output = pkl.load(f)

    baselines = [
        "autofocused-cbas",
        "cbas",
        "bo-qei",
        "cma-es",
        "gradient-ascent",
        "gradient-ascent-min-ensemble",
        "gradient-ascent-mean-ensemble",
        "mins",
        "reinforce"
    ]

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

    baseline_to_scores = {
        "autofocused-cbas": [],
        "cbas": [],
        "bo-qei": [],
        "cma-es": [],
        "gradient-ascent": [],
        "gradient-ascent-min-ensemble": [],
        "gradient-ascent-mean-ensemble": [],
        "mins": [],
        "reinforce": []
    }

    min_num_samples = 128

    for baseline in baselines:

        solutions = glob.glob(f"/home/btrabucco/final-results"
                              f"/{baseline}-nas/*/*/data/solution.npy")

        for soln in solutions:

            data = np.load(soln)
            if data.shape == (128, 64, 4):
                data = task.to_integers(task.denormalize_x(data))

            data = ["-".join(row.tolist()) for row in data]

            baseline_to_scores[baseline].append([
                final_output[row] for row in data if row in final_output])

        print(baseline, ':', baseline_to_scores[baseline])

