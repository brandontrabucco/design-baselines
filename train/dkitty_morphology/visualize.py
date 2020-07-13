import design_bench.factory as fct
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    p = fct.DKittyMorphology(curated=False)

    df = pd.DataFrame.from_dict(
        {
            #"robot": p._robots,
            "score": np.nan_to_num(p._scores[:, 0]),
        }
    )

    df.hist(bins=100, column='score')
    plt.xlabel("Average Return")
    plt.ylabel("Number of Examples")
    plt.title("Coverage of DKitty Morphology")
    plt.savefig('dkitty_dataset_uniform.png')

    p = fct.DKittyMorphology(curated=True)

    df = pd.DataFrame.from_dict(
        {
            #"robot": p._robots,
            "score": np.nan_to_num(p._scores[:, 0]),
        }
    )

    df.hist(bins=100, column='score')
    plt.xlabel("Average Return")
    plt.ylabel("Number of Examples")
    plt.title("Coverage of DKitty Morphology")
    plt.savefig('dkitty_dataset_curated.png')
