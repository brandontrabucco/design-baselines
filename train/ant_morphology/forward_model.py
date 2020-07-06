from morphing_agents.mujoco.ant.elements import LEG
import design_baselines.dev as dev
import design_bench.factory as fct
import pickle as pkl
import os


if __name__ == "__main__":

    designs = []
    for i in range(10):
        alg = dev.ForwardModel(fct.AntMorphology())
        for j in range(10):
            x = alg.solve().cont[0]
            designs.append([
                LEG(*x[:15]), LEG(*x[15:30]), LEG(*x[30:45]), LEG(*x[45:60])])

    os.makedirs("designs/ant/", exist_ok=True)
    with open('designs/ant/forward_model.pkl', 'wb') as f:
        pkl.dump(designs, f)
