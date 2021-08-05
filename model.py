from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from computation_units import Attention, Sparsemax

def flat_accuracy(preds, labels):
    pred_flat = torch.argmax(preds.detach(), axis=1).flatten()
    labels_flat = labels.flatten()
    return torch.sum(pred_flat == labels_flat) / len(labels_flat)

class Paraformer_Model(LightningModule):
    # Set up the classifier
    def __init__(self,base_model="paraphrase-mpnet-base-v2"):
        super().__init__()
        self.plm=SentenceTransformer(base_model)
        self.general_attn=Attention(768)
        self.classifier=nn.Linear(768,2)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, query, article):
        q_vec = self.plm.encode(query,convert_to_tensor=True)
        q_vec = torch.unsqueeze(q_vec,1)
        # print(q_vec.size())
        a_vecs=torch.stack([self.plm.encode(sent,convert_to_tensor=True) for sent in article])
        # print(a_vecs.size())
        a_vecs=a_vecs.permute(1,0,2)
        # print(a_vecs.size())
        attn_output=self.general_attn(q_vec,a_vecs)[0]
        # print(attn_output.size())
        out=self.classifier(attn_output)
        out = torch.squeeze(out,1)
        # print(out.size())
        return out

    def training_step(self, batch, batch_idx):
        b_content, b_article_content, b_article_id, b_labels = batch
        logits = self.forward(b_content, b_article_content)
        
        loss = self.criterion(logits,b_labels)
        acc = flat_accuracy(logits, b_labels)
      
        return {'loss' : loss, 'acc':acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)
        self.log('avg_train_acc', avg_acc)

    def validation_step(self, batch, batch_idx):
        b_content, b_article_content, b_article_id, b_labels = batch
        with torch.no_grad():
          logits = self.forward(b_content, b_article_content)
          
          loss = self.criterion(logits,b_labels)
          acc = flat_accuracy(logits, b_labels)
        
          return {'loss' : loss, 'acc':acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_acc', avg_acc)

    def test_step(self, batch, batch_idx):
        b_content, b_article_content, b_article_id, b_labels = batch
        with torch.no_grad():
          logits = self.forward(b_content, b_article_content)
          
          loss = self.criterion(logits,b_labels)
          acc = flat_accuracy(logits, b_labels)
        
          return {'loss' : loss, 'acc':acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.test_results = {"loss":avg_loss, "acc":avg_acc}
        print('avg_test_loss', avg_loss)
        print('avg_test_acc', avg_acc)
        self.log('avg_test_loss', avg_loss)
        self.log('avg_test_acc', avg_acc)

        
    
    def get_score(self, query, article):
        with torch.no_grad():
          q_vec = self.plm.encode(query,convert_to_tensor=True)
          q_vec = torch.unsqueeze(q_vec,0)#for attention
          q_vec = torch.unsqueeze(q_vec,0)#batch
          a_vecs=torch.stack([self.plm.encode(sent,convert_to_tensor=True) for sent in article])
          a_vecs=torch.unsqueeze(a_vecs,1)#batch
          a_vecs=a_vecs.permute(1,0,2)
          # print(q_vec.size(),a_vecs.size())
          attn_output=self.general_attn(q_vec.cpu().detach(),a_vecs.cpu().detach())[0]
          out=self.classifier(attn_output)
          out = torch.squeeze(out,1)
          # print(out,out.size())
          return out.cpu().detach().numpy()[0][1]


    def predict(self, query, article):
        with torch.no_grad():
          q_vec = self.plm.encode(query,convert_to_tensor=True)
          q_vec = torch.unsqueeze(q_vec,0)#for attention
          q_vec = torch.unsqueeze(q_vec,0)#batch
          a_vecs=torch.stack([self.plm.encode(sent,convert_to_tensor=True) for sent in article])
          a_vecs=torch.unsqueeze(a_vecs,1)#batch
          a_vecs=a_vecs.permute(1,0,2)
          # print(q_vec.size(),a_vecs.size())
          attn_output=self.general_attn(q_vec.detach(),a_vecs.detach())[0]
          out=self.classifier(attn_output)
          out = torch.squeeze(out,1)
          return torch.argmax(out).cpu().detach().numpy()

    # def get_backbone(self):
    #     return self.model

    def configure_optimizers(self):
        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(self.plm.named_parameters())+list(self.general_attn.named_parameters())+list(self.classifier.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.general_attn.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        return torch.optim.Adam(optimizer_grouped_parameters, lr=3e-5,eps=1e-8)