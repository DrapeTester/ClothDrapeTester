import torch
import torch.nn as nn
from torch.nn import functional as F

K = 128
bot_dim = K // 8  # 32

class VAE(nn.Module):
    def __init__(self, feature_dim):
        super(VAE, self).__init__()
        self.feature_dim = feature_dim
        self.en_fc0 = nn.Linear(feature_dim, K)
        self.en_fc1 = nn.Linear(K, K // 2)
        self.en_fc2 = nn.Linear(K // 2, K // 4)
        self.en_fc31 = nn.Linear(K // 4, K // bot_dim)
        self.en_fc32 = nn.Linear(K // 4, K // bot_dim)

        self.de_fc0 = nn.Linear(K // bot_dim, K // 4)
        self.de_fc1 = nn.Linear(K // 4, K // 2)
        self.de_fc2 = nn.Linear(K // 2, K)
        self.de_fc3 = nn.Linear(K, feature_dim)

    def encode(self, x):
        x = F.selu(self.en_fc0(x))
        x = F.selu(self.en_fc1(x))
        x = F.selu(self.en_fc2(x))

        return self.en_fc31(x), self.en_fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.selu(self.de_fc0(z))
        z = F.selu(self.de_fc1(z))
        z = F.selu(self.de_fc2(z))
        z = self.de_fc3(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.feature_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
