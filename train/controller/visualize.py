import design_bench.factory as fct
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    training_dp = fct.ControllerOptimization(
        data_files=("replay_buffer_0_50000.hdf5",
                    "replay_buffer_100000_150000.hdf5"))

    df = pd.DataFrame.from_dict(
        {
            "score": np.nan_to_num(training_dp._traj_return),
        }
    )

    df.hist(bins=100, column='score')
    plt.xlabel("Trajectory Return")
    plt.ylabel("Number of Examples")
    plt.title("Coverage of Hopper Controller Weights")
    plt.savefig('hopper_controller_dataset.png')

    designs = training_dp.sample(n=5)
    print(designs.cont.max(), designs.cont.min())
    s = training_dp.score(designs)

    print(s, designs.score)
