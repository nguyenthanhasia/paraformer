import argparse
from utils import evaluate, load_data_coliee, create_df
from model import Paraformer_Model
from rank_bm25 import BM25Okapi

def weighted_sorted_relevance(query,model,c_docs,c_keys,bm25_top_n=10,alpha=0.1,top_articles=1):
  
  #Get BM25 Scores
  bm25_score=[]
  corpus=list(c_docs)
  tokenized_corpus = [doc.split(" ") for doc in corpus]
  bm25 = BM25Okapi(tokenized_corpus)
  bm25_scores = bm25.get_scores(query.split(" "))

  bm25_filter_list=sorted(list(zip(c_keys,c_docs,bm25_scores)), 
                          key=lambda tup: tup[2],reverse=True)[:bm25_top_n]

  #Get Deep Scores
  c_keys,c_docs,bm25_scores = zip(*bm25_filter_list)
  deep_scores=[]
  final_scores=[]

  #Get Weighted Scores
  for article in bm25_filter_list:
    article_content=[sent.strip() for sent in article[1].split("\n") if sent.strip()!=""]
    deep_score=model.get_score(query, article_content)
    deep_scores.append(deep_score)
    final_scores.append(alpha*deep_score+(1-alpha)*article[2])

  final_list=sorted(list(zip(c_keys,c_docs,bm25_scores,deep_scores,final_scores)), 
                    key=lambda tup: tup[4],reverse=True)[:top_articles]
  
  
  return [article[0] for article in (final_list)]

def main(*args, **kargs):
  print(hyperparams)
  data_dir=hyperparams.data_dir
  test_file=hyperparams.test_file
  ckpt_path=hyperparams.checkpoint
  alpha=hyperparams.alpha
  bm25_top_n=hyperparams.bm25_top_n

  c_docs, c_keys, val_q, test_q, train_q, _ = load_data_coliee(data_dir,test_file=test_file)

  civil_dict={}
  for key,value in zip(c_keys,c_docs):
    civil_dict[key] = value

  df=create_df(test_q,civil_dict,neg_sampling=False)

  model = Paraformer_Model.load_from_checkpoint(checkpoint_path=ckpt_path)
  model.eval()
  # model.cuda()

  df.drop_duplicates(subset =["content","article_id"], keep = "first", inplace = True)
  df = df.groupby('content').article_id.apply(list).reset_index() #group the id

  df["preds"]=df["content"].apply(weighted_sorted_relevance,c_docs=c_docs, c_keys=c_keys,
                                  model=model,bm25_top_n=bm25_top_n,alpha=alpha)
  
  df["true_positive"] = [list(set(df.loc[r, "article_id"]) & set(df.loc[r, "preds"])) for r in range(len(df))]
  df["precision"]=df["true_positive"].apply(len)/df["preds"].apply(len)
  df["recall"]=df["true_positive"].apply(len)/df["article_id"].apply(len)
  df["f2"]=5*df["precision"]*df["recall"]/(4*df["precision"]+df["recall"])
  df["f2"]=df["f2"].fillna(0)
  print(f'Precision: {df["precision"].mean():.3f} | Recall: {df["recall"].mean():.3f} | F2: {df["f2"].mean():.3f}.') 
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--test-file")
    parser.add_argument("--checkpoint")
    parser.add_argument("--bm25-top-n",type=int,default=10)
    parser.add_argument("--alpha",type=float,default=0.1)

    hyperparams = parser.parse_args()

    main(hyperparams)