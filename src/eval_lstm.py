import torch
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_lstm_model(model, dataloader, idx2word, device='cuda' if torch.cuda.is_available() else 'cpu', 
                        num_examples=3):


    model.eval()
    model.to(device)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    all_rouge1 = []
    all_rouge2 = []
    examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Оценка")):
            if batch[0] is None:
                continue
            
            X, Y = batch
            X, Y = X.to(device), Y.to(device)
            
            for i in range(min(len(X), 5)):  
                x = X[i]
                y = Y[i]
                
                # Убираем паддинг
                mask = y != 0
                y = y[mask]
                if len(y) < 2:
                    continue
                
               
                split = int(len(y) * 0.75)
                input_tokens = y[:split]
                target_tokens = y[split:]
                
                if len(target_tokens) == 0:
                    continue
                
                
                gen_indices = model.generate_sequence(
                    input_tokens.cpu().tolist(),
                    max_length=len(target_tokens)
                )
                
                
                gen_tokens = gen_indices[len(input_tokens):]
                
               
                target_words = [idx2word.get(int(idx), '?') for idx in target_tokens.cpu()]
                gen_words = [idx2word.get(int(idx), '?') for idx in gen_tokens if idx != 0]
                
                target_text = ' '.join(target_words)
                gen_text = ' '.join(gen_words)
                
                if len(gen_words) == 0:
                    continue
                
                # rouge
                scores = scorer.score(target_text, gen_text)
                all_rouge1.append(scores['rouge1'].fmeasure)
                all_rouge2.append(scores['rouge2'].fmeasure)
                
               
                if len(examples) < num_examples:
                    input_words = [idx2word.get(int(idx), '?') for idx in input_tokens.cpu()]
                    examples.append({
                        'input': ' '.join(input_words),
                        'generated': gen_text,
                        'target': target_text,
                        'rouge1': scores['rouge1'].fmeasure,
                        'rouge2': scores['rouge2'].fmeasure
                    })
    
    # Усредняем
    metrics = {
        'rouge1': np.mean(all_rouge1) if all_rouge1 else 0.0,
        'rouge2': np.mean(all_rouge2) if all_rouge2 else 0.0
    }
    
    return metrics, examples

def print_examples(examples):
    print("\\n" + "="*60)
    print("Примеры предсказаний")
    print("="*60)
    
    for i, ex in enumerate(examples, 1):
        print(f"\\nПример {i}:")
        print(f"Вход: {ex['input'][:50]}...")
        print(f"Сгенерировано: {ex['generated']}")
        print(f"Target: {ex['target']}")
        print(f"ROUGE-1: {ex['rouge1']:.3f}")
        print(f"ROUGE-2: {ex['rouge2']:.3f}")


