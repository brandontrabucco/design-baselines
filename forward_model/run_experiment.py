from forward_model import ControllerDataset
from forward_model import get_weights
from forward_model import fgsm
from forward_model import step_function
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import pandas as pd
import numpy as np
import os


def run_experiment(config):
    """Train a forward model using various regularization methods and
    solve a model-based optimization problem

    Args:

    config: dict
        a dictionary of hyper parameters for training a forward model
    """

    local_dir = config.get('local_dir', './data')
    init_lr = config.get('init_lr', 0.0001)
    num_epochs = config.get('num_epochs', 100)
    hidden_size = config.get('hidden_size', 2048)
    solver_lr = config.get('solver_lr', 0.1)
    solver_samples = config.get('solver_samples', 10)
    solver_steps = config.get('solver_steps', 201)
    evaluate_interval = config.get('evaluate_interval', 10)
    sc_noise_std = config.get('sc_noise_std', 0.1)
    sc_lambda = config.get('sc_lambda', 10.0)
    sc_weight = config.get('sc_weight', 1.0)
    cs_noise_std = config.get('cs_noise_std', 0.1)
    cs_weight = config.get('cs_weight', 1.0)
    fgsm_lambda = config.get('fgsm_lambda', 0.001)
    fgsm_interval = config.get('fgsm_interval', 1)
    online_noise_std = config.get('online_noise_std', 0.1)
    online_steps = config.get('online_steps', 8)

    tf.io.gfile.makedirs(local_dir)
    data = ControllerDataset()

    model = tf.keras.Sequential([
        tfkl.Dense(hidden_size, use_bias=True,
                   input_shape=(data.robots.shape[1],)),
        tfkl.Activation('relu'),
        tfkl.Dense(hidden_size, use_bias=True),
        tfkl.Activation('relu'),
        tfkl.Dense(1, use_bias=True)])

    loss_df = pd.DataFrame(columns=[
        'Training Iteration',
        'Mean Squared Error',
        'Init LR',
        'Num Epochs',
        'Hidden Size',
        'Solver LR',
        'Solver Samples',
        'Solver Steps',
        'Evaluate Interval',
        'Self-Correcting Noise STD',
        'Self-Correcting Lambda',
        'Self-Correcting Weight',
        'Conservative Noise STD',
        'Conservative Weight',
        'FGSM Lambda',
        'FGSM Interval',
        'Online Noise STD',
        'Online SGD Steps',
        'Type'])

    iteration = 0
    mse = tf.keras.losses.mean_squared_error
    optim = tf.keras.optimizers.Adam(learning_rate=init_lr)

    for epoch in range(num_epochs):
        optim.lr.assign(init_lr * (1 - epoch / num_epochs))
        for X, y, w in data.train:
            loss = step_function(
                optim, model, X, y, w=w[:, 0],
                sc_noise_std=sc_noise_std,
                sc_lambda=sc_lambda,
                sc_weight=sc_weight,
                cs_noise_std=cs_noise_std,
                cs_weight=cs_weight)

            loss_df = loss_df.append({
                'Training Iteration': iteration,
                'Mean Squared Error': loss.numpy(),
                'Init LR': init_lr,
                'Num Epochs': num_epochs,
                'Hidden Size': hidden_size,
                'Solver LR': solver_lr,
                'Solver Samples': solver_samples,
                'Solver Steps': solver_steps,
                'Evaluate Interval': evaluate_interval,
                'Self-Correcting Noise STD': sc_noise_std,
                'Self-Correcting Lambda': sc_lambda,
                'Self-Correcting Weight': sc_weight,
                'Conservative Noise STD': cs_noise_std,
                'Conservative Weight': cs_weight,
                'FGSM Lambda': fgsm_lambda,
                'FGSM Interval': fgsm_interval,
                'Online Noise STD': online_noise_std,
                'Online SGD Steps': online_steps,
                'Type': 'Training'}, ignore_index=True)

            iteration += 1

        loss = 0.0
        num_examples = 0
        for X, y, w in data.val:
            loss += tf.reduce_sum(mse(y, model(X)) * w[:, 0]).numpy()
            num_examples += X.shape[0]
        loss /= num_examples

        print(f"Epoch {epoch} Validation Loss {loss}")

        loss_df = loss_df.append({
            'Training Iteration': iteration,
            'Mean Squared Error': loss,
            'Type': 'Validation'}, ignore_index=True)

        if fgsm_lambda > 0.0 and epoch % fgsm_interval == 0:
            for X, y, w in data.train.take(1):

                # use the fast gradient sign method to find adversarial examples
                x_perturb = X + fgsm_lambda * fgsm(model, X)
                y_perturb = data.score(x_perturb)

                # add the adversarial example to the training set
                data.robots = np.concatenate([
                    data.robots, x_perturb], axis=0).astype(np.float32)
                data.scores = np.concatenate([
                    data.scores, y_perturb], axis=0).astype(np.float32)
                data.weights = get_weights(
                    data.scores).astype(np.float32)

            # rebuild the data set with new samples
            data.build()

    solver_optim = tf.keras.optimizers.Adam(learning_rate=solver_lr)
    x = tf.Variable(data.robots[:solver_samples])

    df = pd.DataFrame(columns=[
        'SGD Steps',
        'Average Return',
        'Init LR',
        'Num Epochs',
        'Hidden Size',
        'Solver LR',
        'Solver Samples',
        'Solver Steps',
        'Evaluate Interval',
        'Self-Correcting Noise STD',
        'Self-Correcting Lambda',
        'Self-Correcting Weight',
        'Conservative Noise STD',
        'Conservative Weight',
        'FGSM Lambda',
        'FGSM Interval',
        'Online Noise STD',
        'Online SGD Steps',
        'Type'])

    # solve the model-based optimization problem
    for n in range(solver_steps):

        # train the forward model by sampling around the current solution
        if online_steps > 0:
            for i in range(online_steps):

                # sample new x around the current solution
                x_online = x + tf.random.normal(x.shape) * online_noise_std
                y_online = data.score(x_online)

                # train the forward model on these new x
                step_function(
                    optim, model, x_online, y_online, w=1,
                    sc_noise_std=sc_noise_std,
                    sc_lambda=sc_lambda,
                    sc_weight=sc_weight,
                    cs_noise_std=cs_noise_std,
                    cs_weight=cs_weight)

        # update the current solution to maximize the forward model
        with tf.GradientTape() as tape:
            predict_s = model(x)[:, 0]
            loss = tf.reduce_sum(-predict_s)

        # evaluate the solution and visualize generalization error
        if n % evaluate_interval == 0:
            real_s = data.score(x)[:, 0]
            for i in range(x.shape[0]):
                df = df.append({
                    'SGD Steps': n,
                    'Average Return': real_s[i],
                    'Init LR': init_lr,
                    'Num Epochs': num_epochs,
                    'Hidden Size': hidden_size,
                    'Solver LR': solver_lr,
                    'Solver Samples': solver_samples,
                    'Solver Steps': solver_steps,
                    'Evaluate Interval': evaluate_interval,
                    'Self-Correcting Noise STD': sc_noise_std,
                    'Self-Correcting Lambda': sc_lambda,
                    'Self-Correcting Weight': sc_weight,
                    'Conservative Noise STD': cs_noise_std,
                    'Conservative Weight': cs_weight,
                    'FGSM Lambda': fgsm_lambda,
                    'FGSM Interval': fgsm_interval,
                    'Online Noise STD': online_noise_std,
                    'Online SGD Steps': online_steps,
                    'Type': 'Oracle'}, ignore_index=True)
                df = df.append({
                    'SGD Steps': n,
                    'Average Return': predict_s[i].numpy(),
                    'Init LR': init_lr,
                    'Num Epochs': num_epochs,
                    'Hidden Size': hidden_size,
                    'Solver LR': solver_lr,
                    'Solver Samples': solver_samples,
                    'Solver Steps': solver_steps,
                    'Evaluate Interval': evaluate_interval,
                    'Self-Correcting Noise STD': sc_noise_std,
                    'Self-Correcting Lambda': sc_lambda,
                    'Self-Correcting Weight': sc_weight,
                    'Conservative Noise STD': cs_noise_std,
                    'Conservative Weight': cs_weight,
                    'FGSM Lambda': fgsm_lambda,
                    'FGSM Interval': fgsm_interval,
                    'Online Noise STD': online_noise_std,
                    'Online SGD Steps': online_steps,
                    'Type': 'FM'}, ignore_index=True)

        grads = tape.gradient(loss, [x])
        solver_optim.apply_gradients(zip(grads, [x]))
        print(f"Solver Step {n}")

    loss_path = os.path.join(local_dir, f'loss.csv')
    eval_path = os.path.join(local_dir, f'eval.csv')
    loss_df.to_csv(loss_path, mode='a', header=not os.path.exists(loss_path))
    df.to_csv(eval_path, mode='a', header=not os.path.exists(eval_path))
