import design_baselines.dev as dev
import design_bench.factory as fct
import numpy as np


if __name__ == "__main__":

    a = dev.MIN(fct.QuadraticOptimization())

    design = a.solve(score=np.array([[0.24]]))
    print(design.cont)

