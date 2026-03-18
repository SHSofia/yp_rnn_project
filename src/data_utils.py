import pandas as pd
import re
import urllib.request
import zipfile
import os
import shutil

def download_and_save_raw_data():
    print("Загрузка датасета sentiment140")
    
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    
    os.makedirs("data/temp", exist_ok=True)
    zip_path = "data/temp/sentiment140.zip"
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/temp/")
    
    extracted_files = os.listdir("data/temp/")
    csv_files = [f for f in extracted_files if f.endswith('.csv')]
    print(f"Найдено CSV файлов: {csv_files}")
    
    # Объединяем все CSV файлы
    all_dfs = []
    for csv_file in csv_files:
        csv_path = os.path.join("data/temp", csv_file)
        df = pd.read_csv(csv_path, 
                         encoding='latin-1', 
                         header=None,
                         names=['target', 'ids', 'date', 'flag', 'user', 'text'],
                         usecols=['text'])
        all_dfs.append(df)
        print(f"Файл {csv_file}: {len(df)} записей")
    
    # Объединяем все данные
    df = pd.concat(all_dfs, ignore_index=True)
    
    output_path = "data/raw_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Сырой датасет сохранен в {output_path}")
    print(f"Всего записей: {len(df)}")
    
    shutil.rmtree("data/temp")
    return df


def clean_text(text):

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_and_tokenize_data(df):

    df['cleaned_text'] = df['text'].apply(clean_text)
    
    df['tokenized_text'] = df['cleaned_text'].apply(lambda x: x.split())
    
    df = df[df['tokenized_text'].apply(lambda x: len(x) > 0)]
    
    output_path = "data/dataset_processed.csv"
    df_to_save = df[['cleaned_text', 'tokenized_text']].copy()
    df_to_save['tokenized_text'] = df_to_save['tokenized_text'].apply(lambda x: ' '.join(x))
    df_to_save.to_csv(output_path, index=False)
    
    return df
