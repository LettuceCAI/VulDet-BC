import torch 
import torch.nn as nn 
from torch.nn import functional as F

class BVTS_BiGRU_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gru_size, filter_num, dropout, weights=None, is_pretrain=False):
        super(BVTS_BiGRU_CNN, self).__init__()
        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim)

        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)
        self.word_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)
        self.word_fc = nn.Linear(2*gru_size, 2*gru_size)

        self.sentence_gru = nn.GRU(input_size=2*gru_size, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)

        # self.class_fc = nn.Linear(2*gru_size, class_num)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = filter_num, 
                                              kernel_size = (fs, 2*gru_size)) 
                                    for fs in [2,4,6,8,10,12,16,20]
                                    ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(8*filter_num, 2)

        nn.init.xavier_normal_(self.word_query, gain=1)

    def forward(self, x):
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)
        embed_x = self.word_embed(x)
        word_output, _ = self.word_gru(embed_x)

        word_attention = torch.tanh(self.word_fc(word_output))

        weights = torch.matmul(word_attention, self.word_query)
        weights = F.softmax(weights, dim=1)

        x = x.unsqueeze(2)
        weights = torch.where(x!=1, weights, torch.full_like(x, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        
        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sentence_num, word_output.size(2))

        sentence_output, _ = self.sentence_gru(sentence_vector)
        text = sentence_output.unsqueeze(1)
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)

class Con_BiGRU_CNN(nn.Module):
    def __init__(self, gru_size, filter_num):
        super(Con_BiGRU_CNN, self).__init__()

        self.sentence_gru = nn.GRU(input_size=160, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)

        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = filter_num, 
                                              kernel_size = (fs, 2*gru_size)) 
                                    for fs in [2,4,6,8,10,12,16,20]
                                    ])
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(8*filter_num, 2)

    def forward(self, x):
        x = x.float()
        x, _ = self.sentence_gru(x)
        text = x.unsqueeze(1)
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)
    
class onlyAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gru_size, filter_num, dropout, weights=None, is_pretrain=False):
        super(onlyAttention, self).__init__()
        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim)

        self.word_query = nn.Parameter(torch.Tensor(embedding_dim, 1), requires_grad=True)
        self.word_fc = nn.Linear(embedding_dim, embedding_dim)

        self.sentence_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)

        # self.class_fc = nn.Linear(2*gru_size, class_num)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = filter_num, 
                                              kernel_size = (fs, 2*gru_size)) 
                                    for fs in [2,4,6,8,10,12,16,20]
                                    ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(8*filter_num, 2)

        nn.init.xavier_normal_(self.word_query, gain=1)

    def forward(self, x):
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)
        embed_x = self.word_embed(x)
        word_attention = torch.tanh(self.word_fc(embed_x))

        weights = torch.matmul(word_attention, self.word_query)
        weights = F.softmax(weights, dim=1)

        x = x.unsqueeze(2)
        weights = torch.where(x!=257, weights, torch.full_like(x, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        
        sentence_vector = torch.sum(embed_x, dim=1).view(-1, sentence_num, embed_x.size(2))

        sentence_output, _ = self.sentence_gru(sentence_vector)
        text = sentence_output.unsqueeze(1)
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)

class BiGRUwithoutAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gru_size, filter_num, dropout, weights=None, is_pretrain=False):
        super(BiGRUwithoutAttention, self).__init__()
        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim)

        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)

        self.sentence_gru = nn.GRU(input_size=2*gru_size, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)

        # self.class_fc = nn.Linear(2*gru_size, class_num)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = filter_num, 
                                              kernel_size = (fs, 2*gru_size)) 
                                    for fs in [2,4,6,8,10,12,16,20]
                                    ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(8*filter_num, 2)


    def forward(self, x):
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)
        embed_x = self.word_embed(x)
        word_output, _ = self.word_gru(embed_x)
        
        sentence_vector = torch.sum(word_output, dim=1).view(-1, sentence_num, word_output.size(2))

        sentence_output, _ = self.sentence_gru(sentence_vector)
        text = sentence_output.unsqueeze(1)
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)
    
class BVTS_BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gru_size, filter_num, dropout, weights=None, is_pretrain=False):
        super(BVTS_BiGRU, self).__init__()
        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim)

        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)
        self.word_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)
        self.word_fc = nn.Linear(2*gru_size, 2*gru_size)

        self.sentence_gru = nn.GRU(input_size=2*gru_size, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*gru_size, 2)

        nn.init.xavier_normal_(self.word_query, gain=1)

    def forward(self, x):
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)
        embed_x = self.word_embed(x)
        word_output, _ = self.word_gru(embed_x)

        word_attention = torch.tanh(self.word_fc(word_output))

        weights = torch.matmul(word_attention, self.word_query)
        weights = F.softmax(weights, dim=1)

        x = x.unsqueeze(2)
        weights = torch.where(x!=257, weights, torch.full_like(x, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        
        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sentence_num, word_output.size(2))
        _, gru_hidden = self.sentence_gru(sentence_vector)
        x1 = []
        for i in range(gru_hidden.size(0)):
            x1.append(gru_hidden[i, :, :])
        x1 = torch.cat(x1, dim=-1)

        cat = self.dropout(x1)
        return self.fc(cat)
    
class BVTS_BiGRU_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gru_size, filter_num, dropout, class_num, weights=None, is_pretrain=False):
        super(BVTS_BiGRU_CNN, self).__init__()
        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim)

        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)
        self.word_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)
        self.word_fc = nn.Linear(2*gru_size, 2*gru_size)

        self.sentence_gru = nn.GRU(input_size=2*gru_size, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)

        # self.class_fc = nn.Linear(2*gru_size, class_num)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = filter_num, 
                                              kernel_size = (fs, 2*gru_size)) 
                                    for fs in [2,4,6,8,10,12,16,20]
                                    ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(8*filter_num, 2)

        nn.init.xavier_normal_(self.word_query, gain=1)

    def forward(self, x):
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)
        embed_x = self.word_embed(x)
        word_output, _ = self.word_gru(embed_x)

        word_attention = torch.tanh(self.word_fc(word_output))

        weights = torch.matmul(word_attention, self.word_query)
        weights = F.softmax(weights, dim=1)

        x = x.unsqueeze(2)
        weights = torch.where(x!=257, weights, torch.full_like(x, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        
        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sentence_num, word_output.size(2))

        sentence_output, _ = self.sentence_gru(sentence_vector)
        text = sentence_output.unsqueeze(1)
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)
    
class BVTS_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gru_size, filter_num, dropout, weights=None, is_pretrain=False):
        super(BVTS_CNN, self).__init__()
        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim)

        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)
        self.word_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)
        self.word_fc = nn.Linear(2*gru_size, 2*gru_size)

        self.sentence_gru = nn.GRU(input_size=2*gru_size, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)

        # self.class_fc = nn.Linear(2*gru_size, class_num)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = filter_num, 
                                              kernel_size = (fs, 2*gru_size)) 
                                    for fs in [2,4,6,8,10,12,16,20]
                                    ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(8*filter_num, 2)

        nn.init.xavier_normal_(self.word_query, gain=1)

    def forward(self, x):
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)
        embed_x = self.word_embed(x)
        word_output, _ = self.word_gru(embed_x)

        word_attention = torch.tanh(self.word_fc(word_output))

        weights = torch.matmul(word_attention, self.word_query)
        weights = F.softmax(weights, dim=1)

        x = x.unsqueeze(2)
        weights = torch.where(x!=257, weights, torch.full_like(x, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sentence_num, word_output.size(2))

        text = sentence_vector.unsqueeze(1)
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)
    
class BVTS_CNN_BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gru_size, filter_num, dropout, weights=None, is_pretrain=False):
        super(BVTS_CNN_BiGRU, self).__init__()
        if is_pretrain:
            self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embedding_dim)

        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)
        self.word_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)
        self.word_fc = nn.Linear(2*gru_size, 2*gru_size)

        self.sentence_gru = nn.GRU(input_size=filter_num, hidden_size=gru_size, num_layers=1, bidirectional=True, batch_first=True)

        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = filter_num, 
                                              kernel_size = (fs, 2*gru_size)) 
                                    for fs in [2,4,6,8,10,12,16,20]
                                    ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*8*gru_size, 2)

        nn.init.xavier_normal_(self.word_query, gain=1)

    def forward(self, x):
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)
        embed_x = self.word_embed(x)
        word_output, _ = self.word_gru(embed_x)

        word_attention = torch.tanh(self.word_fc(word_output))

        weights = torch.matmul(word_attention, self.word_query)
        weights = F.softmax(weights, dim=1)

        x = x.unsqueeze(2)
        weights = torch.where(x!=257, weights, torch.full_like(x, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        
        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sentence_num, word_output.size(2))

        sentence_vector = sentence_vector.unsqueeze(1)
        conved = [F.relu(conv(sentence_vector)).squeeze(3) for conv in self.convs]
        gru_hid = [self.sentence_gru(con.transpose(1, 2))[1] for con in conved]
        inputs = []
        for out in gru_hid:
            x = []
            for i in range(out.size(0)):
                x.append(out[i, :, :])
            x = torch.cat(x, dim=-1)
            inputs.append(x)
        cat = self.dropout(torch.cat(inputs, dim = -1))
        return self.fc(cat)

class HAN_BSVD(nn.Module):
    def __init__(self, embedding_dim=128, gru_size=256, weights=None):
        super(HAN_BSVD, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(weights, freeze=False)

        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size, num_layers=3, bidirectional=True, batch_first=True)
        self.word_query = nn.Parameter(torch.Tensor(2*gru_size, 1), requires_grad=True)
        self.word_fc = nn.Linear(2*gru_size, 2*gru_size)

        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = 256, 
                                              kernel_size = (fs, 2*gru_size)) 
                                    for fs in [2,4,6,8,10,12,16,20]
                                    ])
        
        self.convs1 = nn.ModuleList([
                                    nn.Conv2d(in_channels = 2, 
                                              out_channels = 1, 
                                              kernel_size = (int(((150 - fs + 1) + 2) / 2), 1)) # int(((150 - fs + 1) + 2) / 2)
                                    for fs in [2,4,6,8,10,12,16,20]
                                    ])
        
        self.sigmoid = nn.Sigmoid()
        self.line = nn.ModuleList([nn.Linear(int(((150 - fs + 1) + 2) / 2), 150 - fs + 1)for fs in [2,4,6,8,10,12,16,20]])
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(8*256, 2)
        self.softmax = nn.Softmax(dim=1)

        nn.init.xavier_normal_(self.word_query, gain=1)

    def forward(self, x):
        sentence_num = x.size(1)
        sentence_len = x.size(2)
        x = x.view(-1, sentence_len)
        embed_x = self.word_embed(x)
        word_output, _ = self.word_gru(embed_x)
        word_attention = torch.relu(self.word_fc(word_output))
        weights = torch.matmul(word_attention, self.word_query)
        weights = F.softmax(weights, dim=1)

        x = x.unsqueeze(2)
        weights = torch.where(x!=0, weights, torch.full_like(x, 0, dtype=torch.float))
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        

        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sentence_num, word_output.size(2))
        text = sentence_vector.unsqueeze(1)
        conved = [conv(text).squeeze(3) for conv in self.convs]
        conout = []
        i = 0
        for x in conved:
            x_tran = x.transpose(1, 2)
            x_avg = F.avg_pool1d(x_tran, x_tran.shape[2]).transpose(1, 2)
            x_max = F.max_pool1d(x_tran, x_tran.shape[2]).transpose(1, 2)
            x_con = torch.cat([x_avg, x_max], dim = 1)
            x_con = x_con.unsqueeze(3)
            spital = self.convs1[i](x_con)
            spital = self.sigmoid(spital)
            spital = self.line[i](spital.squeeze(3))
            spital = self.sigmoid(spital)
            x_out = torch.mul(x, spital)
            x_out = F.max_pool1d(x_out, x_out.shape[2]).squeeze(2)
            conout.append(x_out)
            i += 1
        cat = self.dropout(torch.cat(conout, dim = 1))
        out =  self.softmax(self.fc(cat))
        return out