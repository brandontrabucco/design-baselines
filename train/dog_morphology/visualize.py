import design_bench.factory as fct
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    p = fct.DogMorphology(centered=True)

    df = pd.DataFrame.from_dict(
        {
            "score": np.nan_to_num(p._scores[:, 0]),
        }
    )

    df.hist(bins=100, column='score')
    plt.xlabel("Average Return")
    plt.ylabel("Number of Examples")
    plt.title("Coverage of Dog Morphology")
    plt.savefig('dog_dataset_curated.png')

    p = fct.DogMorphology(centered=False)

    df = pd.DataFrame.from_dict(
        {
            "score": np.nan_to_num(p._scores[:, 0]),
        }
    )

    df.hist(bins=100, column='score')
    plt.xlabel("Average Return")
    plt.ylabel("Number of Examples")
    plt.title("Coverage of Dog Morphology")
    plt.savefig('dog_dataset_uniform.png')

