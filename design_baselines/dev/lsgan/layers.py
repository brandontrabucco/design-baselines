import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


class DenseConditionalGenerator(nn.Module):

    def __init__(self, latent_size, hidden_size, output_size,
                 num_embeddings, embedding_size):
        super(DenseConditionalGenerator, self).__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_size)
        self.model = nn.Sequential(
            nn.Linear(latent_size + embedding_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh())

    def forward(self, z, y):
        return self.model(torch.cat([z, self.emb(y)], dim=1))


class DenseConditionalDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_size,
                 num_embeddings, embedding_size):
        super(DenseConditionalDiscriminator, self).__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_size)
        self.model = nn.Sequential(
            nn.Linear(input_size + embedding_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_size, 1))

    def forward(self, x, y):
        return self.model(torch.cat([x, self.emb(y)], dim=1))


class ConvConditionalGenerator(nn.Module):

    def __init__(self, latent_size, hidden_size, output_size,
                 num_embeddings, embedding_size):
        super(ConvConditionalGenerator, self).__init__()
        self.latent_size = latent_size
        self.embedding_size = embedding_size
        self.emb = nn.Embedding(num_embeddings, 2 * 2 * embedding_size)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size + embedding_size,
                               hidden_size,
                               5,
                               stride=2,
                               padding=2,
                               output_padding=1),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size,
                               hidden_size,
                               13,
                               stride=7,
                               padding=6,
                               output_padding=6),
            nn.InstanceNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, output_size, 3, padding=1),
            nn.Tanh())

    def forward(self, z, y):
        return self.model(torch.cat([
            z.reshape(-1, self.latent_size, 2, 2),
            self.emb(y).reshape(-1, self.embedding_size, 2, 2)], dim=1))


class ConvConditionalDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_size,
                 num_embeddings, embedding_size):
        super(ConvConditionalDiscriminator, self).__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_size)
        self.out = nn.Linear(2 * 2 * hidden_size + embedding_size,
                             embedding_size)
        self.model = nn.Sequential(
            nn.Conv2d(input_size,
                      hidden_size,
                      13,
                      stride=7,
                      padding=6),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_size,
                      hidden_size,
                      5,
                      stride=2,
                      padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.Flatten())

    def forward(self, x, y):
        return self.out(torch.cat([
            self.model(x), self.emb(y)], dim=1))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    data = tv.datasets.MNIST("./", download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=4)

    G = ConvConditionalGenerator(64, 256, 1, 10, 64)
    D = ConvConditionalDiscriminator(1, 256, 10, 64)

    G.cuda()
    D.cuda()

    G.train()
    D.train()

    G_optim = torch.optim.Adam(G.parameters(), lr=2e-5, betas=(0.5, 0.999))
    D_optim = torch.optim.Adam(D.parameters(), lr=2e-5, betas=(0.5, 0.999))

    for e in range(15):
    
        for i, (img, y) in enumerate(data_loader):

            img = img.cuda() * 2.0 - 1.0
            y = y.cuda()

            d_img = D(img, y).mean()

            real_p = 0.5 * ((D(img, y) - 1)**2).mean()

            fake_img = G(torch.randn(img.shape[0], 64, 2, 2).cuda(), y).detach()

            d_fake_img = D(fake_img, y).mean()

            fake_p = 0.5 * (D(fake_img, y)**2).mean()

            D_loss = real_p + fake_p

            D.zero_grad()
            D_loss.backward()
            D_optim.step()

            fake_img = G(torch.randn(img.shape[0], 64, 2, 2).cuda(), y)

            fake_p = 0.5 * ((D(fake_img, y) - 1)**2).mean()

            G_loss = fake_p

            G.zero_grad()
            G_loss.backward()
            G_optim.step()

            print(f"e {e} : i {i} : G Loss {G_loss} : D Loss {D_loss} : D(x) {d_img} : D(G(z)) {d_fake_img}")

    for n in range(3):

        x = fake_img[n].detach().cpu().numpy().reshape(28, 28) / 2 + 0.5
        plt.imshow(x)
        plt.show()

        x = img[n].detach().cpu().numpy().reshape(28, 28) / 2 + 0.5
        plt.imshow(x)
        plt.show()

