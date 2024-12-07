import wget
import os
url = 'https://raw.githubusercontent.com/wandb/edu/main/mlops-001/lesson1/requirements.txt'
filename = 'requirements.txt'
if not os.path.exists(filename):
    wget.download(url, filename)

import gdown
url = 'https://drive.google.com/file/d/1Ns80_G8fAA0xs4e-7yvCVToYD-D5bNmb/view?usp=sharing'
output = 'params.py'
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

import wandb
import shutil
import pandas as pd
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

import params

train_config = SimpleNamespace(
    project=params.WANDB_PROJECT,
    entity=params.ENTITY,
    bdd_classes=params.BDD_CLASSES,
    raw_data_at=params.RAW_DATA_AT,
    processed_data_at=params.PROCESSED_DATA_AT,
    solver='saga',
    max_iter=50,
    class_weight='balanced',
    log_preds=True,
    job_type="train",
    framework="sklearn",
    seed=42,
)

def download_data():
  processed_data_at = wandb.use_artifact(f'{train_config.processed_data_at}:latest')
  processed_dataset_dir = Path(processed_data_at.download())
  df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
  return df

def preprocess_data():
  df = download_data()
  df_copy = df.copy()
  label_encoder = LabelEncoder()
  df_copy['classe'] = label_encoder.fit_transform(df['classe'])

  return df_copy

def get_df(is_test=False):
  df = preprocess_data()

  if is_test:
      return df[df['Stage'] == 'test']
  else:
      df_train = df[df['Stage'] == 'train']
      df_valid = df[df['Stage'] == 'valid']
      return df_train, df_valid

def split(is_test=False):

  if not is_test:
      df_train, df_valid = get_df(is_test=False)
      X_train = df_train.drop(columns=['classe', '0', 'Stage'])
      y_train = df_train['classe']
      X_val = df_valid.drop(columns=['classe', '0', 'Stage'])
      y_val = df_valid['classe']
      return X_train, y_train, X_val, y_val
  else:
      df_test = get_df(is_test=True)
      X_test = df_test.drop(columns=['classe', '0', 'Stage'])
      y_test = df_test['classe']
      return X_test, y_test

def log_predictions(val_accuracy, val_report, test_accuracy, test_report, y_test, y_test_pred):

  print("==== Resultados - Validação ====")
  print(f"Acurácia: {val_accuracy:.4f}")
  print(val_report)

  print("\n==== Resultados - Teste ====")
  print(f"Acurácia: {test_accuracy:.4f}")
  print(test_report)

  df_test = get_df(is_test=True)
  predictions_df = pd.DataFrame({
      'id': df_test['0'],
      'Ground Truth': y_test,
      'Predictions': y_test_pred
  })

  # Criar uma tabela W&B
  table = wandb.Table(dataframe=predictions_df)

  # Logar a tabela no W&B
  wandb.log({"prediction_table": table})

def log_final_metrics(val_accuracy, test_accuracy):
    # Salvar as métricas finais
    final_metrics = {
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy
    }

    # Logar as métricas finais no W&B
    wandb.summary.update(final_metrics)
    print("Final metrics logged.")
def train_model(train_config):
    run = wandb.init(project=train_config.project, entity=train_config.entity, job_type=train_config.job_type, config=train_config)

    # Acessando os parâmetros do sweep diretamente do config
    config = wandb.config
    log_preds = config.log_preds
    solver = config.solver
    max_iter = config.max_iter
    class_weight = config.class_weight

    X_train, y_train, X_val, y_val = split()
    X_test, y_test = split(is_test=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Criando o modelo com os parâmetros do sweep
    model = LogisticRegression(
        random_state=train_config.seed, 
        max_iter=max_iter, 
        solver=solver, 
        class_weight=class_weight
    )

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)

    if log_preds:
        log_predictions(val_accuracy, val_report, test_accuracy, test_report, y_test, y_test_pred)
    log_final_metrics(val_accuracy, test_accuracy)
    
    shutil.copy('train.py', wandb.run.dir)
    
    wandb.finish()


if __name__ == '__main__':
    train_model(train_config)