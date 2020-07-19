from collections import defaultdict
import design_baselines.dev as dev
import design_bench.factory as fct
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")


STEPS = list(range(0, 200, 10))


if __name__ == "__main__":

    name_to_scores = defaultdict(list)
    df = pd.DataFrame(columns=['SGD Steps',
                               'Average Return',
                               'Type'])

    training_dp = fct.ControllerOptimization(
        data_files=("replay_buffer_0_50000.hdf5",
                    "replay_buffer_100000_150000.hdf5"))

    validation_dp = fct.ControllerOptimization(
        data_files=("replay_buffer_50000_100000.hdf5",
                    "replay_buffer_150000_200000.hdf5"))

    variants = {
        "Lambda = 100": dict(
            num_layers=2,
            hidden_size=512,
            batch_size=512,
            training_iterations=1000,
            init_lr=0.0001,
            num_sgd_steps=10,
            discrete_size=1000,
            init_from_dataset=True,
            conservative_noise_std=0.5,
            conservative_lambda=50.0,
            conservative_weight=1.0,
            add_noise=False,
            noise_std=0.01,
            label_interpolation=False),
        "Lambda = 10": dict(
            num_layers=2,
            hidden_size=512,
            batch_size=512,
            training_iterations=1000,
            init_lr=0.0001,
            num_sgd_steps=10,
            discrete_size=1000,
            init_from_dataset=True,
            conservative_noise_std=0.5,
            conservative_lambda=10.0,
            conservative_weight=1.0,
            add_noise=False,
            noise_std=0.01,
            label_interpolation=False),
        "Lambda = 1": dict(
            num_layers=2,
            hidden_size=512,
            batch_size=512,
            training_iterations=1000,
            init_lr=0.0001,
            num_sgd_steps=10,
            discrete_size=1000,
            init_from_dataset=True,
            conservative_noise_std=0.5,
            conservative_lambda=1.0,
            conservative_weight=1.0,
            add_noise=False,
            noise_std=0.01,
            label_interpolation=False),
        "Original": dict(
            num_layers=2,
            hidden_size=512,
            batch_size=512,
            training_iterations=1000,
            init_lr=0.0001,
            num_sgd_steps=10,
            discrete_size=1000,
            init_from_dataset=True,
            conservative_noise_std=0.0,
            conservative_lambda=0.0,
            conservative_weight=0.0,
            add_noise=False,
            noise_std=0.01,
            label_interpolation=False),
    }

    for name, kwargs in variants.items():

        alg = dev.ForwardModel(training_dp, validation_dp, **kwargs)

        plt.clf()
        plt.plot(alg.training_losses, label="Training")
        plt.plot(alg.validation_losses, label="Validation")
        plt.title("Model Loss")
        plt.ylabel("Logcosh Loss")
        plt.xlabel("Training Iteration")
        plt.legend()
        plt.savefig(f"losses_controller_fm_{name}.png")

        plt.clf()
        plt.plot(alg.training_rms, label="Training Unexplained")
        plt.plot(alg.training_std, label="Training Total")
        plt.plot(alg.validation_rms, label="Validation Unexplained")
        plt.plot(alg.validation_std, label="Validation Total")
        plt.title("Model Error")
        plt.ylabel("Standard Deviation")
        plt.xlabel("Training Iteration")
        plt.legend()
        plt.savefig(f"std_controller_fm_{name}.png")

        for n in STEPS:
            alg.num_sgd_steps = n
            design = alg.solve(n=10)
            score = training_dp.score(design)
            for i in range(10):
                df = df.append({'SGD Steps': n,
                                'Average Return': score[i, 0],
                                'Type': name},
                               ignore_index=True)
                df = df.append({'SGD Steps': n,
                                'Average Return': design.score[i, 0],
                                'Type': name + " FM"},
                               ignore_index=True)

    plt.clf()
    sns.lineplot(x="SGD Steps", y="Average Return", hue='Type', data=df)
    plt.title("Hopper Controller Optimization")
    plt.savefig("controller_fm_conservative.png")
