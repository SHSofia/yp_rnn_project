import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class NextTokenDataset(Dataset):
    
    def __init__(self, tokenized_texts, max_length=60):

        self.tokenized_texts = tokenized_texts
        self.max_length = max_length
        

        self._build_vocabulary()
        
        # Преобразуем все тексты в индексы
        self.indexed_texts = []
        for text in tokenized_texts:
            if len(text) > max_length:
                text = text[:max_length]

            indexed = [self.word2idx[word] for word in text]
            self.indexed_texts.append(indexed)
    
    def _build_vocabulary(self):
        # Собираем все уникальные слова
        all_words = set()
        for text in self.tokenized_texts:
            all_words.update(text)
        
        # Сортируем 
        all_words = sorted(list(all_words))
        
        # Создаем словари 
        self.word2idx = {}
        self.idx2word = {}
        
        for i, word in enumerate(all_words):
            self.word2idx[word] = i
            self.idx2word[i] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Размер словаря: {self.vocab_size}")
    
    def __len__(self):
        return len(self.indexed_texts)
    
    def __getitem__(self, idx):
        text = self.indexed_texts[idx]
        
        if len(text) < 2:
            return None
        
        X = text[:-1]  
        Y = text[1:] 
        
        return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

def collate_fn(batch):

    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None, None

    X_batch, Y_batch = zip(*batch)
    

    max_len = max([len(x) for x in X_batch])
    
    # Паддинг
    X_padded = []
    Y_padded = []
    
    for x, y in zip(X_batch, Y_batch):
        pad_len = max_len - len(x)
        
        if pad_len > 0:
            x_padded = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y_padded = torch.cat([y, torch.zeros(pad_len, dtype=torch.long)])
        else:
            x_padded = x
            y_padded = y
        
        X_padded.append(x_padded)
        Y_padded.append(y_padded)
    
    return torch.stack(X_padded), torch.stack(Y_padded)

def create_dataloaders(data_path="data/dataset_processed.csv", 
                      batch_size=256,
                      max_length=50,
                      test_size=0.2,
                      val_size=0.1):

    df = pd.read_csv(data_path)
    
    tokenized_texts = df['tokenized_text'].apply(lambda x: x.split()).tolist()

    tokenized_texts = [text for text in tokenized_texts if len(text) >= 2]

    train_texts, test_texts = train_test_split(
        tokenized_texts, 
        test_size=test_size, 
        random_state=42
    )
    
    train_texts, val_texts = train_test_split(
        train_texts,
        test_size=val_size/(1-test_size),
        random_state=42
    )
    
    print(f"train: {len(train_texts)}")
    print(f"val: {len(val_texts)}")
    print(f"tet: {len(test_texts)}")

    # Создаем датасеты
    train_dataset = NextTokenDataset(train_texts, max_length)
    val_dataset = NextTokenDataset(val_texts, max_length)
    test_dataset = NextTokenDataset(test_texts, max_length)
    
    # Создаем даталоадеры
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset.vocab_size, train_dataset.idx2word