import networkx
import numpy as np
import torch.optim as optimizer
from torch.utils.data.dataloader import DataLoader
import model as MMN
from torch.utils import data
import torch


def read_graph(file_name):
    edge = np.loadtxt(file_name).astype(np.int32)
    min_node, max_node = edge.min(), edge.max()
    if min_node == 0:
        Node = max_node + 1
    else:
        Node = max_node
    G = networkx.Graph()
    Adj = np.zeros([Node, Node], dtype=np.int32)
    for i in range(edge.shape[0]):
        G.add_edge(edge[i][0], edge[i][1])
        if min_node == 0:
            Adj[edge[i][0], edge[i][1]] = 1
            Adj[edge[i][1], edge[i][0]] = 1
        else:
            Adj[edge[i][0] - 1, edge[i][1] - 1] = 1
            Adj[edge[i][1] - 1, edge[i][0] - 1] = 1
    Adj = torch.FloatTensor(Adj)
    return G, Adj, Node


class Dataload(data.Dataset):
    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.Node


def train_SDNE(model, data, adj_matrix, opt, epochs, nu1=1e-5, nu2=1e-4, beta=5, step_size=10, gamma=0.9):
    scheduler = optimizer.lr_scheduler.StepLR(
        opt, step_size=step_size, gamma=gamma)
    model.train()
    for epoch in range(1, epochs + 1):
        loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
        for index in data:
            adj_batch = adj_matrix[index]
            adj_mat = adj_batch[:, index]
            b_mat = torch.ones_like(adj_batch)
            b_mat[adj_batch != 0] = beta

            opt.zero_grad()
            L_1st, L_2nd, L_all = model(adj_batch, adj_mat, b_mat)
            L_reg = 0
            for param in model.parameters():
                L_reg += nu1 * \
                    torch.sum(torch.abs(param)) + nu2 * \
                    torch.sum(param * param)
            Loss = L_all + L_reg
            Loss.backward()
            opt.step()
            loss_sum += Loss
            loss_L1 += L_1st
            loss_L2 += L_2nd
            loss_reg += L_reg
        scheduler.step(epoch)
        # print("The lr for epoch %d is %f" %(epoch, scheduler.get_lr()[0]))
        print("Epoch %d:" % epoch)
        print("+ loss_sum is %f" % loss_sum)
        print("+ loss_L1 is %f" % loss_L1)
        print("+ loss_L2 is %f" % loss_L2)
        print("+ loss_reg is %f" % loss_reg)

    return model


def main():
    nhid0 = 1000
    nhid1 = 128
    dropout = 0.5
    alpha = 1e-2
    learning_rate = 0.001
    batch_size = 20
    shuffle = True
    epochs = 15
    data_path = "data/data.edgelist"
    embedding_path = "data/embedding.txt"

    G, Adj, Node = read_graph(data_path)
    model = MMN.MNN(Node, nhid0, nhid1, dropout, alpha)

    data = Dataload(Adj, Node)
    data = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    optimize = optimizer.Adam(model.parameters(), lr=learning_rate)
    model.to("cpu")

    model = train_SDNE(model=model, data=data, adj_matrix=Adj,
                       opt=optimize, epochs=epochs)

    model.eval()
    embedding = model.save_vector(Adj)
    embedding_vector = embedding.detach().numpy()
    np.savetxt(embedding_path, embedding_vector)


if __name__ == '__main__':
    main()
