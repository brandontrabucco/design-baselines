from collections import defaultdict
import design_baselines.dev as dev
import design_bench.factory as fct
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")


STEPS = list(range(0, 20, 1)) + \
        list(range(20, 40, 2)) + \
        list(range(40, 100, 4)) + \
        list(range(100, 200, 10))


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
        "Lambda = 1": dict(
            num_layers=3,
            hidden_size=2048,
            batch_size=512,
            training_iterations=1000,
            init_lr=0.001,
            num_sgd_steps=10,
            discrete_size=1000,
            init_from_dataset=True,
            conservative_noise_std=0.1,
            conservative_lambda=1.0,
            conservative_weight=1.0,
            add_noise=False,
            noise_std=0.01,
            label_interpolation=False),
        "Lambda = 100": dict(
            num_layers=3,
            hidden_size=2048,
            batch_size=512,
            training_iterations=1000,
            init_lr=0.001,
            num_sgd_steps=10,
            discrete_size=1000,
            init_from_dataset=True,
            conservative_noise_std=0.1,
            conservative_lambda=100.0,
            conservative_weight=1.0,
            add_noise=False,
            noise_std=0.01,
            label_interpolation=False),
        "Lambda = 10": dict(
            num_layers=3,
            hidden_size=2048,
            batch_size=512,
            training_iterations=1000,
            init_lr=0.001,
            num_sgd_steps=10,
            discrete_size=1000,
            init_from_dataset=True,
            conservative_noise_std=0.1,
            conservative_lambda=10.0,
            conservative_weight=1.0,
            add_noise=False,
            noise_std=0.01,
            label_interpolation=False),
        "Original": dict(
            num_layers=3,
            hidden_size=2048,
            batch_size=512,
            training_iterations=1000,
            init_lr=0.001,
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
        plt.plot(alg.training_rms, label="Training Model Error")
        plt.plot(alg.training_std, label="Training Dataset")
        plt.plot(alg.validation_rms, label="Validation Model Error")
        plt.plot(alg.validation_std, label="Validation Dataset")
        plt.title("Model Error")
        plt.ylabel("Standard Deviation")
        plt.xlabel("Training Iteration")
        plt.legend()
        plt.savefig(f"std_controller_fm_{name}.png")

        for n in STEPS:
            for i in range(10):
                alg.num_sgd_steps = n
                design = alg.solve()
                score = training_dp.score(design)[0, 0]
                df = df.append({'SGD Steps': n,
                                'Average Return': score,
                                'Type': name},
                               ignore_index=True)

    plt.clf()
    sns.lineplot(x="SGD Steps", y="Average Return", hue='Type', data=df)
    plt.title("Hopper Controller Optimization")
    plt.savefig("controller_fm_conservative.png")
