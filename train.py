import argparse
from utils import evaluate, load_data_coliee, create_df
import pandas as pd
from data import Paraformer_DataModule
from pytorch_lightning import Trainer
from model import Paraformer_Model


def main(*args, **kargs):
  data_dir=hyperparams.data_dir
  test_file=hyperparams.test_file
  max_epochs=hyperparams.max_epochs
  save_top_k=hyperparams.save_top_k
  patience=hyperparams.patience
  gpus=hyperparams.gpus


  c_docs, c_keys, val_q, test_q, train_q, _ = load_data_coliee(data_dir,test_file=test_file)

  civil_dict={}
  for key,value in zip(c_keys,c_docs):
    civil_dict[key] = value

  df_train=create_df(train_q,civil_dict)
  df_val=create_df(val_q,civil_dict)
  df_test=create_df(test_q,civil_dict,neg_sampling=False)

  data_module=Paraformer_DataModule(df_train,df_val,df_test)
  data_module.setup()

  from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
  checkpoint_callback = ModelCheckpoint(
      monitor='avg_val_loss',
      filename='Paraformer',
      save_top_k=save_top_k, #  save the top 3 models
      mode='min', # mode of the monitored quantity  for optimization
  )
  from pytorch_lightning.callbacks import EarlyStopping
  early_stop_callback = EarlyStopping(
    monitor='avg_val_loss',
    min_delta=0.00,
    patience=patience,
    verbose=False,
    mode='min'
  )

  trainer = Trainer(max_epochs = max_epochs , gpus =gpus, callbacks=[checkpoint_callback,early_stop_callback])

  model=Paraformer_Model()
  trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--test-file")
    parser.add_argument("--max-epochs",type=int,default=200)
    parser.add_argument("--save-top-k",type=int,default=3)
    parser.add_argument("--patience",type=int,default=20)
    parser.add_argument("--gpus",type=int,default=1)

    hyperparams = parser.parse_args()

    main(hyperparams)