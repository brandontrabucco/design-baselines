from morphing_agents.mujoco.dog.elements import LEG
import design_baselines.dev as dev
import design_bench.factory as fct
import pickle as pkl
import os
import numpy as np


if __name__ == "__main__":

    designs = []
    for i in range(3):
        alg = dev.LSGAN(fct.DogMorphology())
        for j in range(3):
            x = alg.solve(score=np.array([[1000.0]])).cont[0]
            designs.append([
                LEG(*x[:14]), LEG(*x[14:28]), LEG(*x[28:42]), LEG(*x[42:56])])

    os.makedirs("designs/dog/", exist_ok=True)
    with open('designs/dog/lsgan.pkl', 'wb') as f:
        pkl.dump(designs, f)
