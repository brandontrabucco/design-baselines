import design_baselines.dev as dev
import design_bench.factory as fct
import tensorflow as tf


if __name__ == "__main__":

    a = dev.ForwardModel(fct.DKittyMorphology())

    design = a.solve()
    print(design.cont)

