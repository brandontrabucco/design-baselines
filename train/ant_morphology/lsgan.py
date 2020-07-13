from morphing_agents.mujoco.ant.elements import LEG
import design_baselines.dev as dev
import design_bench.factory as fct
import pickle as pkl
import os
import numpy as np


if __name__ == "__main__":

    designs = []
    for i in range(3):
        alg = dev.LSGAN(fct.AntMorphology())
        for j in range(1):
            x = alg.solve(score=np.array([[1000.0]])).cont[0]
            designs.append([
                LEG(*x[:15]), LEG(*x[15:30]), LEG(*x[30:45]), LEG(*x[45:60])])

    os.makedirs("designs/ant/", exist_ok=True)
    with open('designs/ant/lsgan.pkl', 'wb') as f:
        pkl.dump(designs, f)
