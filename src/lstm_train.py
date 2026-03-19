import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.lstm_model import LSTMAutocomplete
from src.eval_lstm import evaluate_lstm_model  

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Обучение")
    for batch in pbar:
        if batch[0] is None:
            continue
        
        X, Y = batch
        X, Y = X.to(device), Y.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches

def train_model(train_loader, val_loader, vocab_size, idx2word, 
                embedding_dim=128, hidden_dim=128, num_layers=2,
                learning_rate=0.001, num_epochs=5, device='cuda' if torch.cuda.is_available() else 'cpu',
                model_save_path='models/lstm_model.pth'):
    
    print(f"Используемое устройство: {device}")
    
    model = LSTMAutocomplete(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_losses = []
    val_rouge1_scores = []
    val_rouge2_scores = []
    
    print(f"Начинаем тренировку на {num_epochs} эпох...")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Эпоха {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        print("Оценка на валидационной выборке...")
        val_metrics, examples = evaluate_lstm_model(
            model, 
            val_loader, 
            idx2word, 
            device=device,
            num_examples=3
        )
        
        val_rouge1_scores.append(val_metrics['rouge1'])
        val_rouge2_scores.append(val_metrics['rouge2'])
        
        print(f"\nРезультаты эпохи {epoch + 1}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val ROUGE-1: {val_metrics['rouge1']:.4f}")
        print(f"Val ROUGE-2: {val_metrics['rouge2']:.4f}")
        
        # Вместо print_examples просто выведем примеры вручную
        if examples:
            print("\nПримеры:")
            for i, ex in enumerate(examples, 1):
                print(f"{i}. Вход: {ex['input'][:30]}... -> {ex['generated']}")
        
        if epoch == 0 or val_metrics['rouge1'] > max(val_rouge1_scores[:-1]):
            print(f"✓ Сохраняем лучшую модель (ROUGE-1: {val_metrics['rouge1']:.4f})")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_rouge1': val_metrics['rouge1'],
                'val_rouge2': val_metrics['rouge2'],
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers
            }, model_save_path)
    
    # Рисуем графики
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses)
    plt.title('Train Loss')
    plt.subplot(1,2,2)
    plt.plot(val_rouge1_scores, label='ROUGE-1')
    plt.plot(val_rouge2_scores, label='ROUGE-2')
    plt.title('Validation ROUGE')
    plt.legend()
    plt.savefig('models/training_history.png')
    plt.show()
    
    print(f"\n{'='*50}")
    print("ТРЕНИРОВКА ЗАВЕРШЕНА")
    print(f"{'='*50}")
    print(f"Лучшая модель сохранена в {model_save_path}")
    print(f"Лучший ROUGE-1: {max(val_rouge1_scores):.4f}")
    print(f"Лучший ROUGE-2: {max(val_rouge2_scores):.4f}")
    
    return model, train_losses, val_rouge1_scores, val_rouge2_scores