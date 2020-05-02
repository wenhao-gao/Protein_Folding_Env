import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, DataLoader, Batch
import ipdb


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0., std=0.1)
        torch.nn.init.constant_(m.bias, 0.1)


class Net(torch.nn.Module):
    def __init__(self, args, device):
        super(Net, self).__init__()


        self.args = args
        self.device = device

        node_dim = self.args.node_dim
        edge_dim = self.args.edge_dim
        hidden_dim = self.args.hidden_dim
        processing_steps = self.args.processing_steps
        self.depth = self.args.depth

        self.lin0 = torch.nn.Linear(node_dim, hidden_dim)
        nn = Sequential(Linear(edge_dim, hidden_dim * 2), ReLU(), Linear(hidden_dim * 2, hidden_dim * hidden_dim))
        self.conv = NNConv(hidden_dim, hidden_dim, nn, aggr='mean')
        self.gru = GRU(hidden_dim, hidden_dim)

        self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps)
        self.lin1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 1)

        self.lin3 = torch.nn.Linear(hidden_dim, 36)
        self.lin4 = torch.nn.Linear(36, 2)

        self.lin5 = torch.nn.Linear(hidden_dim, 36)
        self.lin6 = torch.nn.Linear(36, 2)

        self.apply(init_weights)

    def forward(self, data):
        data, out, batch_size = self.encode(data)
        data.to(self.device)
        value = self.critic(out, data.batch)
        probs = self.actor_proba(out, batch_size)
        dist = Categorical(probs)
        return dist, value

    def encode(self, data):

        if isinstance(data, Data):
            data_list = [data]
            loader = DataLoader(data_list, batch_size=1)
            for data in loader:
                break
        elif isinstance(data, list):
            loader = DataLoader(data, batch_size=len(data))
            for data in loader:
                break
        elif isinstance(data, Batch):
            pass

        batch_size = data.batch.max().item() + 1

        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.depth):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        self.embedding = out
        self.batch_size = batch_size

        return data, out, batch_size

    def critic(self, data, batch):
        data = self.set2set(data, batch)
        data = F.relu(self.lin1(data))
        data = self.lin2(data)
        return data #.view(-1)

    def actor_proba(self, data, batch_size):
        data = F.relu(self.lin3(data))
        data = F.softmax(self.lin4(data).view(batch_size, -1))
        return data

    def get_value(self, action, std=0.01):
        data = self.embedding
        data.to(self.device)
        batch_size = self.batch_size
        data = F.relu(self.lin5(data))
        data = torch.tanh(self.lin6(data)).view(batch_size, -1)
        mu = torch.stack([data[i][action[i]] for i in range(len(action))], dim=0).view(batch_size, )
        log_std = torch.nn.Parameter(torch.ones(1, ) * std)
        std = log_std.exp().expand_as(mu)
        mu = mu.to(self.device)
        std = std.to(self.device)
        dist = Normal(mu, std)
        return dist
