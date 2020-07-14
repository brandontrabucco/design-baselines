from morphing_agents.mujoco.dog.elements import LEG
import design_baselines.dev as dev
import design_bench.factory as fct
import pickle as pkl
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    designs = []
    for i in range(3):

        # there are 100 samples, hold out 20
        training_dp = fct.DogMorphology(curated=True)
        training_dp._robots = training_dp._robots[20:]
        training_dp._scores = training_dp._scores[20:]

        # these 20 samples become the validation set
        validation_dp = fct.DogMorphology(curated=True)
        validation_dp._robots = validation_dp._robots[:20]
        validation_dp._scores = validation_dp._scores[:20]

        alg = dev.ForwardModel(training_dp,
                               validation_dp,
                               num_layers=2,
                               hidden_size=512,
                               batch_size=32,
                               training_iterations=10000,
                               init_lr=0.0001,
                               num_sgd_steps=10,
                               discrete_size=1000,
                               init_from_dataset=True)

        plt.clf()
        plt.plot(alg.training_losses, label="Training")
        plt.plot(alg.validation_losses, label="Validation")
        plt.title("Model Loss")
        plt.ylabel("Logcosh Loss")
        plt.xlabel("Training Iteration")
        plt.legend()
        plt.savefig(f"losses_{i}_dog_forward_model.png")

        plt.clf()
        plt.plot(alg.training_rms, label="Training Model Error")
        plt.plot(alg.training_std, label="Training Dataset")
        plt.plot(alg.validation_rms, label="Validation Model Error")
        plt.plot(alg.validation_std, label="Validation Dataset")
        plt.title("Model Error")
        plt.ylabel("Standard Deviation")
        plt.xlabel("Training Iteration")
        plt.legend()
        plt.savefig(f"std_{i}_dog_forward_model.png")

        for j in range(3):
            x = alg.solve().cont[0]
            designs.append([
                LEG(*x[:14]), LEG(*x[14:28]), LEG(*x[28:42]), LEG(*x[42:56])])

    os.makedirs("designs/dog/", exist_ok=True)
    with open('designs/dog/forward_model.pkl', 'wb') as f:
        pkl.dump(designs, f)
