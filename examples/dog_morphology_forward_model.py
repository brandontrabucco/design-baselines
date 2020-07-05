import design_baselines.dev as dev
import design_bench.factory as fct
import pickle as pkl
import numpy as np
import os


if __name__ == "__main__":

    designs = []
    for i in range(10):
        a = dev.ForwardModel(fct.DogMorphology())
        for j in range(10):
            designs.append(a.solve())

    designs = np.array(designs)
    os.makedirs("designs/dog/", exist_ok=True)
    np.save("designs/dog/forward_model.npy", designs)
