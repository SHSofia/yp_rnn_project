import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rouge_score import rouge_scorer

class TransformerEvaluator:
    def __init__(self, model_name="distilgpt2", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Загрузка {model_name} на {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = device
        print("Готово!")
    
    def generate_completion(self, text, max_new_tokens=10):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )
        
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
    
    def evaluate_on_dataset(self, texts, split_ratio=0.75, max_new_tokens=10, num_examples=5):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        all_rouge1 = []
        all_rouge2 = []
        examples = []
        
        for text in tqdm(texts, desc="Оценка"):
            words = text.split()
            if len(words) < 4:
                continue
            
            split = int(len(words) * split_ratio)
            input_text = ' '.join(words[:split])
            target_text = ' '.join(words[split:])
            
            if len(target_text.split()) == 0:
                continue
            
            try:
                generated = self.generate_completion(input_text, max_new_tokens=len(words[split:]))
                
                scores = scorer.score(target_text, generated)
                all_rouge1.append(scores['rouge1'].fmeasure)
                all_rouge2.append(scores['rouge2'].fmeasure)
                
                if len(examples) < num_examples:
                    examples.append({
                        'input': input_text[:50] + '...' if len(input_text) > 50 else input_text,
                        'generated': generated,
                        'target': target_text,
                        'rouge1': scores['rouge1'].fmeasure,
                        'rouge2': scores['rouge2'].fmeasure
                    })
            except:
                continue
        
        metrics = {
            'rouge1': np.mean(all_rouge1) if all_rouge1 else 0.0,
            'rouge2': np.mean(all_rouge2) if all_rouge2 else 0.0
        }
        
        return metrics, examples

def compare_with_lstm(transformer_metrics, transformer_examples, lstm_metrics, lstm_examples):
    print("\n" + "="*50)
    print("Сравнение моделей")
    print("="*50)
    print(f"{'Модель':<15} {'ROUGE-1':<10} {'ROUGE-2':<10}")
    print("-"*35)
    print(f"{'LSTM':<15} {lstm_metrics['rouge1']:.4f}    {lstm_metrics['rouge2']:.4f}")
    print(f"{'DistilGPT2':<15} {transformer_metrics['rouge1']:.4f}    {transformer_metrics['rouge2']:.4f}")
    
    print("\nТрансформер. Пример:")
    for i, ex in enumerate(transformer_examples, 1):
        print(f"\n{i}. Вход: {ex['input']}")
        print(f"   Ген: {ex['generated']}")
        print(f"   Цель: {ex['target']}")