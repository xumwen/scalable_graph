import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size, hidden_size)
    
    def forward(self, X):
        """
        :param X: Input data of shape (input_seq_len, batch_size, hidden_size).
        :return output: Output data of shape (input_seq_len, batch_size, hidden_size).
        :return hidden: Last hidden of shape (1, batch_size, hidden_size)
        """
        hidden = torch.zeros(1, X.shape[1], self.hidden_size, device=X.device)
        output, hidden = self.gru(X, hidden)
        return output, hidden


class EncoderMultiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes, num_rnns=6):
        super(EncoderMultiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_rnns = num_rnns
        self.grus = [torch.nn.GRU(input_size, hidden_size) for i in range(num_rnns)]
        self.embed = nn.Parameter(torch.FloatTensor(num_nodes,num_rnns))
        self.reset_parameters()

    def forward(self, X):
        """
        :param X: Input data of shape (input_seq_len, batch_size, hidden_size).
        :return output: Output data of shape (input_seq_len, batch_size, hidden_size).
        :return hidden: Last hidden of shape (1, batch_size, hidden_size)
        """
        gru_output = torch.zeros(X.shape[0],X.shape[1],self.hidden_size,self.num_rnns,device=X.device)
        for i in range(self.num_rnns):
            self.grus[i].to(device=X.device)
            t, h = self.grus[i].forward(X)
            gru_output[:,:,:,i] = t
        
        gru_output = gru_output.contiguous().view(X.shape[0],-1,self.num_nodes,self.hidden_size,self.num_rnns)
        output = torch.einsum("ijkmn,kn->ijkm",[gru_output, torch.softmax(self.embed, dim=-1)])
        output = output.contiguous().view(X.shape[0],X.shape[1],self.hidden_size)
        return output, output[-1,:,:]

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.embed.shape[1])
        self.embed.data.uniform_(-stdv1, stdv1)


class DecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_seq_len):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
     
    def forward(self, hidden):
        """
        :param hidden: Encoder's last hidden of shape (batch_size, hidden_size).
        :return: Output data of shape (batch_size, output_seq_len).
        """
        output = torch.zeros(hidden.shape[0], self.output_seq_len, device=hidden.device)
        for i in range(self.output_seq_len):
            x = self.linear(hidden)[:,0]
            hidden = self.gru_cell(hidden, hidden)
            output[:,i] = x
        
        return output
    
class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, input_seq_len, output_seq_len):
        super(GRUBlock, self).__init__()
        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len
        self.encoder_gru = EncoderGRU(input_size, hidden_size)
        self.decoder_gru = DecoderGRU(hidden_size, hidden_size, output_seq_len)
        
    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, input_seq_len,
        num_features).
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # batch_size * num_nodes -> batch
        gru_input = X.contiguous().view(-1, X.shape[2], X.shape[3]).permute(1, 0, 2)
        t, hidden = self.encoder_gru(gru_input)
        output = self.decoder_gru(hidden[0])
        output = output.contiguous().view(X.shape[0], X.shape[1], self.output_seq_len)  
        # batch -> batch_size * num_nodes 
        return output