from design_baselines.core import Algorithm
from design_bench.core import DesignProblem, Design, Container
import numpy as np
import torch
import torch.nn as nn


class DenseConditionalGenerator(nn.Module):

    def __init__(self, latent_size, hidden_size, output_size):
        super(DenseConditionalGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size + 1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh())

    def forward(self, z, y):
        return self.model(torch.cat([z, y], dim=1))


class DenseConditionalDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(DenseConditionalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size + 1, hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_size, 1))

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


class LSGAN(Algorithm):
    """
    A general interface for model-based optimization algorithms
    For example, one might use a forward model with gradient descent

    Usage:

    p = DesignProblem(...)
    algorithm = Algorithm(p, ...)
    design = algorithm.solve(...)
    """

    def __init__(self,
                 design_problem: DesignProblem,
                 hidden_size=256,
                 batch_size=32,
                 training_iterations=20000,
                 latent_dim=32):
        """
        Create a general interface for optimizations algorithms that solve
        model-based optimization problems

        Arguments:

        design_problem: DesignProblem
            A specific design problem that will be solved using the algorithm
            implemented in this class
        """

        Algorithm.__init__(self,
                           design_problem,
                           hidden_size=hidden_size,
                           batch_size=batch_size,
                           training_iterations=training_iterations,
                           latent_dim=latent_dim)

        assert self.design_problem.is_continuous

        # sample bounds for the generator
        self.shift = torch.FloatTensor(
            self.design_problem.design_space.upper[np.newaxis] +
            self.design_problem.design_space.lower[np.newaxis]).cuda() / 2
        self.scale = torch.FloatTensor(
            self.design_problem.design_space.upper[np.newaxis] -
            self.design_problem.design_space.lower[np.newaxis]).cuda() / 2

        self.G = DenseConditionalGenerator(latent_dim, hidden_size, self.shift.shape[1])
        self.D = DenseConditionalDiscriminator(self.shift.shape[1], hidden_size)

        self.G.cuda()
        self.D.cuda()

        self.G.train()
        self.D.train()

        self.G_optim = torch.optim.Adam(
            self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(
            self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Train

        for i in range(training_iterations):
            design = self.design_problem.sample(n=batch_size)
            design.score = np.nan_to_num(design.score)

            x = torch.FloatTensor(design.cont).cuda()
            y = torch.FloatTensor(design.score).cuda()

            d_real = self.D(x, y).mean()

            real_p = 0.5 * ((self.D(x, y) - 1) ** 2).mean()

            fake_x = self.G(torch.randn(x.shape[0], self.latent_dim).cuda(), y)
            fake_x = fake_x * self.scale + self.shift

            d_fake = self.D(fake_x, y).mean()

            fake_p = 0.5 * (self.D(fake_x, y) ** 2).mean()

            D_loss = real_p + fake_p

            self.D.zero_grad()
            D_loss.backward()
            self.D_optim.step()

            fake_x = self.G(torch.randn(x.shape[0], self.latent_dim).cuda(), y)
            fake_x = fake_x * self.scale + self.shift

            fake_p = 0.5 * ((self.D(fake_x, y) - 1) ** 2).mean()

            G_loss = fake_p

            self.G.zero_grad()
            G_loss.backward()
            self.G_optim.step()

            print(f"i {i} : G Loss {G_loss} : D Loss {D_loss} : "
                  f"D(x) {d_real} : D(G(z)) {d_fake}")

    def solve(self,
              score: np.ndarray = None,
              condition: Container = None) -> Design:
        """
        Solve the design problem by conditioning on inputs and finding a
        design that maximizes the design problem score

        Arguments:

        inputs: Any
            An optimal input used to parameterize the design problem
            For example, conditioning on an attribute like color
        """

        assert score is not None

        self.G.eval()

        x = torch.randn(score.shape[0], self.latent_dim).cuda()
        y = torch.FloatTensor(score).cuda()
        sample = self.G(x, y) * self.scale + self.shift

        design = self.design_problem.design_space.sample(n=1)
        design.cont = sample.cpu().detach().numpy()
        design.score = score
        design.condition = condition
        return design
