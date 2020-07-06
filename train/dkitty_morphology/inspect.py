import design_bench.factory as fct
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from softlearning.environments.gym.mujoco.morphing_dkitty import MorphingDKittyWalkFixed, Leg


if __name__ == "__main__":

    p = fct.DKittyMorphology()

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
    plt.savefig('dkitty_dataset.png')

    x = p._robots[np.argmax(np.nan_to_num(p._scores[:, 0]))]
    print(x.tolist())

    legs = [
        Leg(*x[:14]),
        Leg(*x[14:28]),
        Leg(*x[28:42]),
        Leg(*x[42:56]),
    ]

    frames = []

    e = MorphingDKittyWalkFixed(legs=legs)
    e.reset()
    for i in range(100):
        e.step(e.action_space.sample())
        img = e.render(mode='rgb_array')
        frames.append(img)

    import skvideo.io
    skvideo.io.vwrite("best_dkitty.mp4", frames)

