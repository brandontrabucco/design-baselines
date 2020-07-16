import design_bench.factory as fct
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    p = fct.ControllerOptimization()

    df = pd.DataFrame.from_dict(
        {
            "score": np.nan_to_num(p._traj_return),
        }
    )

    df.hist(bins=100, column='score')
    plt.xlabel("Trajectory Return")
    plt.ylabel("Number of Examples")
    plt.title("Coverage of Hopper Controller Weights")
    plt.savefig('hopper_controller_dataset.png')

    designs = p.sample(n=5)
    s = p.score(designs)

    print(s, designs.score)
