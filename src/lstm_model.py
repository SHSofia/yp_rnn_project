import torch
import torch.nn as nn

class LSTMAutocomplete(nn.Module):

    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        
        logits = self.fc(output)
        
        return logits, hidden
    
    def generate_next_token(self, x, hidden=None):
        logits, hidden = self.forward(x, hidden)
        last_token_logits = logits[:, -1, :]
        next_token_idx = torch.argmax(last_token_logits, dim=-1)
        return next_token_idx, hidden
    
    def generate_sequence(self, start_tokens, max_length=20, idx2word=None):
        self.eval()
        
        if isinstance(start_tokens, list):
            start_tokens = torch.tensor([start_tokens])
        
        if start_tokens.dim() == 1:
            start_tokens = start_tokens.unsqueeze(0)
        
        batch_size = start_tokens.size(0)
        current_tokens = start_tokens
        hidden = None
        
        all_tokens = start_tokens[0].tolist() if batch_size == 1 else start_tokens.tolist()
        
        for _ in range(max_length):
            last_token = current_tokens[:, -1:]
            next_token, hidden = self.generate_next_token(last_token, hidden)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)
            
            if batch_size == 1:
                all_tokens.append(next_token.item())
        
        if idx2word is not None and batch_size == 1:
            words = [idx2word.get(idx, '<UNK>') for idx in all_tokens]
            return words, all_tokens
        
        return all_tokens