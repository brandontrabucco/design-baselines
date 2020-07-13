import design_bench.factory as fct
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    p = fct.AntMorphology(curated=True)

    df = pd.DataFrame.from_dict(
        {
            #"robot": p._robots,
            "score": np.nan_to_num(p._scores[:, 0]),
        }
    )

    df.hist(bins=100, column='score')
    plt.xlabel("Average Return")
    plt.ylabel("Number of Examples")
    plt.title("Coverage of Ant Morphology")
    plt.savefig('ant_dataset_curated.png')

    p = fct.AntMorphology(curated=False)

    df = pd.DataFrame.from_dict(
        {
            #"robot": p._robots,
            "score": np.nan_to_num(p._scores[:, 0]),
        }
    )

    df.hist(bins=100, column='score')
    plt.xlabel("Average Return")
    plt.ylabel("Number of Examples")
    plt.title("Coverage of Ant Morphology")
    plt.savefig('ant_dataset_uniform.png')
