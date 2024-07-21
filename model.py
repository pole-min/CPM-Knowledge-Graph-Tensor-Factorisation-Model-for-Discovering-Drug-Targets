import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d1)
        # self.R = torch.nn.Embedding(len(d.relations), d2)
        # self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
        #                             dtype=torch.float, device="cuda", requires_grad=True))

        W = np.random.uniform(0, 0, (d1, d1, d1))
        for i in range(d1):
            for j in range(d1):
                for k in range(d1):
                    if i == j == k:
                        W[i][j][k] = 1
                    else:
                        W[i][j][k] = 0
        self.W = torch.nn.Parameter(torch.tensor(W, dtype=torch.float, device="cuda", requires_grad=False), requires_grad=False)

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    # def forward(self, e1_idx, r_idx):
    #     e1 = self.E(e1_idx)
    #     x = self.bn0(e1)
    #     x = self.input_dropout(x)
    #     x = x.view(-1, 1, e1.size(1))
    #
    #     r = self.R(r_idx)
    #     W_mat = torch.mm(r, self.W.view(r.size(1), -1))
    #     W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
    #     W_mat = self.hidden_dropout1(W_mat)
    #
    #     x = torch.bmm(x, W_mat)  # (128,1,200)
    #     # print(x.size())
    #     x = x.view(-1, e1.size(1))
    #     x = self.bn1(x)
    #     x = self.hidden_dropout2(x)
    #     x = torch.mm(x, self.E.weight.transpose(1,0))
    #     pred = torch.sigmoid(x)
    #     return pred

    def forward(self, e1_idx, r_idx):
        r = self.R(r_idx)
        y = r.view(-1, 1, r.size(1))
        # print(y.size()) #([128, 1, 200])

        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        W_mat = torch.mm(x, self.W.view(x.size(1), -1))
        W_mat = W_mat.view(-1, r.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)
        # print(W_mat.size()) #([128,200,200])

        x = torch.bmm(y, W_mat)
        # print(x.size())  #([128, 1, 200])
        # x = torch.mm(r,W_mat.view(r.size(1),-1))
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        # print(x.size()) #([128, 200])
        # print( self.E.weight.transpose(1,0).size()) #([200, 40943])
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred
