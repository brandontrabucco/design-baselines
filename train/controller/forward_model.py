from collections import defaultdict
import design_baselines.dev as dev
import design_bench.factory as fct
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")


STEPS = [0, 1, 2, 5, 10, 20]


if __name__ == "__main__":

    name_to_scores = defaultdict(list)
    df = pd.DataFrame(columns=['SGD Steps', 'Average Return'])

    designs = []
    for i in range(3):

        training_dp = fct.ControllerOptimization(
            data_files=("replay_buffer_0_50000.hdf5",
                        "replay_buffer_100000_150000.hdf5"))

        validation_dp = fct.ControllerOptimization(
            data_files=("replay_buffer_50000_100000.hdf5",
                        "replay_buffer_150000_200000.hdf5"))

        alg = dev.ForwardModel(training_dp,
                               validation_dp,
                               num_layers=2,
                               hidden_size=512,
                               batch_size=512,
                               training_iterations=10,
                               init_lr=0.001,
                               num_sgd_steps=10,
                               discrete_size=1000,
                               init_from_dataset=True,
                               add_noise=False,
                               label_interpolation=False)

        plt.clf()
        plt.plot(alg.training_losses, label="Training")
        plt.plot(alg.validation_losses, label="Validation")
        plt.title("Model Loss")
        plt.ylabel("Logcosh Loss")
        plt.xlabel("Training Iteration")
        plt.legend()
        plt.savefig(f"losses_{i}_controller_forward_model.png")

        plt.clf()
        plt.plot(alg.training_rms, label="Training Model Error")
        plt.plot(alg.training_std, label="Training Dataset")
        plt.plot(alg.validation_rms, label="Validation Model Error")
        plt.plot(alg.validation_std, label="Validation Dataset")
        plt.title("Model Error")
        plt.ylabel("Standard Deviation")
        plt.xlabel("Training Iteration")
        plt.legend()
        plt.savefig(f"std_{i}_controller_forward_model.png")

        for n in STEPS:
            for j in range(3):
                design = alg.solve()
                score = training_dp.score(design)[0, 0]
                df = df.append({'SGD Steps': n, 'Average Return': score},
                               ignore_index=True)

    sns.lineplot(x="SGD Steps", y="Average Return", data=df)
    plt.title("Hopper Controller Optimization")
    plt.savefig("controller_forward_model.png")
