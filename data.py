from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
from pytorch_lightning import LightningDataModule
BATCH_SIZE=1

class Paraformer_Dataset(Dataset):
    def __init__(self, df):
      self.content = df["content"]
      self.article_content = df["article_content"]
      self.article_id = df["article_id"]
      self.label = df["label"]

    def __getitem__(self, idx):
      return self.content[idx], self.article_content[idx], self.article_id[idx],self.label[idx]
          
    def __len__(self):
      return len(self.content)


class Paraformer_DataModule(LightningDataModule):
    
  def __init__(self,df_train=None, df_val=None, df_test=None):
    super().__init__()
    self.df_train, self.df_val, self.df_test=df_train, df_val, df_test

  def setup(self):
    if self.df_train is not None:
      self.train_dataset = Paraformer_Dataset(self.df_train)
    if self.df_val is not None:
      self.val_dataset= Paraformer_Dataset(self.df_val)
    if self.df_test is not None:
      self.test_dataset=Paraformer_Dataset(self.df_test)
        
        
  def train_dataloader(self):
    return DataLoader(self.train_dataset,batch_size=BATCH_SIZE, shuffle = True, num_workers=4)

  def val_dataloader(self):
    return DataLoader(self.val_dataset,batch_size= BATCH_SIZE, shuffle = False, num_workers=4)

  def test_dataloader(self):
    return DataLoader(self.test_dataset,batch_size= BATCH_SIZE, shuffle = False, num_workers=4)