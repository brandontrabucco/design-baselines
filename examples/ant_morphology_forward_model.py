import design_baselines.dev as dev
import design_bench.factory as fct
import pickle as pkl
import numpy as np
import os


if __name__ == "__main__":

    designs = []
    for i in range(10):
        a = dev.ForwardModel(fct.AntMorphology())
        for j in range(10):
            designs.append(a.solve())

    designs = np.array(designs)
    os.makedirs("designs/ant/", exist_ok=True)
    np.save("designs/ant/forward_model.npy", designs)
