import os
import sys
from tqdm import tqdm
import time
import argparse

from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from src.CLER.utils import *
from src.CLER.model import *
from src.CLER.dataset import GTDatasetWithLabel, SingleEntityDataset

def get_topK(embeddingA, embeddingB, topK = 100, hp=None):
    embeddingA = torch.tensor(np.array(embeddingA)).cuda()
    embeddingB = torch.tensor(np.array(embeddingB)).cuda()
    # First check whether we have already computed the topK before
    if hp is not None:
        embeddings_path = os.path.join('models', hp.dataset +'_CLER_{}'.format(hp.total_budget) + '_seed_{}'.format(hp.run_id))
        if os.path.exists(os.path.join(embeddings_path, 'topkA.txt')) and embeddingA.shape[0] > 1000:
            topkA = np.loadtxt(os.path.join(embeddings_path, 'topkA.txt'), delimiter=',')
            distA = np.loadtxt(os.path.join(embeddings_path, 'distA.txt'), delimiter=',')
            return topkA, distA, ''

    if (embeddingA.shape[0] > 1000 or embeddingB.shape[0] > 1000) and hp is not None:
        # The embeddings are too large, we need to split the similarity computation into batches
        batch_size = 256
        for batch_idx in tqdm(range(embeddingA.shape[0]//batch_size), desc='Computing topK', total=embeddingA.shape[0]//batch_size):
            sim_score_batch = util.pytorch_cos_sim(embeddingA[batch_idx*batch_size:(batch_idx+1)*batch_size], embeddingB)
            distA_batch, topkA_batch = torch.topk(sim_score_batch, k=topK, dim=1)
            # We write the results to csv files in order to avoid memory issues
            with open(os.path.join(embeddings_path, 'topkA.txt'), 'a') as f:
                np.savetxt(f, topkA_batch.cpu().numpy(), delimiter=',')
            with open(os.path.join(embeddings_path, 'distA.txt'), 'a') as f:
                np.savetxt(f, distA_batch.cpu().numpy(), delimiter=',')

        if embeddingA.shape[0] % batch_size != 0:
            sim_score_batch = util.pytorch_cos_sim(embeddingA[(batch_idx+1)*batch_size:], embeddingB)
            distA_batch, topkA_batch = torch.topk(sim_score_batch, k=topK, dim=1)
            # We write the results to files in order to avoid memory issues
            with open(os.path.join(embeddings_path, 'topkA.txt'), 'a') as f:
                np.savetxt(f, topkA_batch.cpu().numpy(), delimiter=',')
            with open(os.path.join(embeddings_path, 'distA.txt'), 'a') as f:
                np.savetxt(f, distA_batch.cpu().numpy(), delimiter=',')


        # Read the results from the files
        topkA = np.loadtxt(os.path.join(embeddings_path, 'topkA.txt'), delimiter=',')
        distA = np.loadtxt(os.path.join(embeddings_path, 'distA.txt'), delimiter=',')
        return topkA, distA, ''

    else:
        sim_score = util.pytorch_cos_sim(embeddingA, embeddingB)
        distA, topkA = torch.topk(sim_score, k=topK, dim=1) # topkA [sizeA, K]
        topkA = topkA.cpu().numpy()
        distA = distA.cpu().numpy()
        sim_score = sim_score.cpu().numpy()
        return topkA, distA, sim_score

def get_emb(hp, load_test=False):
    emb_path = os.path.join('models', hp.dataset +'_CLER_{}'.format(hp.total_budget) + '_seed_{}'.format(hp.run_id), 'embeddingA.pt')
    if os.path.exists(emb_path) and load_test:
        embeddingA = torch.load(emb_path)
        if 'companies' in hp.dataset:
            return embeddingA, embeddingA
        else:
            embeddingB = torch.load(os.path.join('models', hp.dataset +'_CLER_{}'.format(hp.total_budget) + '_seed_{}'.format(hp.run_id), 'embeddingB.pt'))
            return embeddingA, embeddingB

    path = hp.path
    if load_test:
        attr_listA, entity_listA, attr_listB, entity_listB = load_attributes(path, load_test=True)
    else:
        attr_listA, entity_listA, attr_listB, entity_listB = load_attributes(path)

    datasetA = SingleEntityDataset(entity_listA, attr_listA, lm='sent-bert', max_len=128, add_token=hp.add_token)
    datasetB = SingleEntityDataset(entity_listB, attr_listB, lm='sent-bert', max_len=128, add_token=hp.add_token)
    blocker = CLSepModel(lm='sent-bert')
    blocker = blocker.cuda()
    bk_opt = AdamW(blocker.parameters(), lr=hp.lr)
    if hp.ckpt_type == 'best':
        blocker.load_state_dict(torch.load(os.path.join('models', hp.dataset +'_CLER_{}'.format(hp.total_budget) + '_seed_{}'.format(hp.run_id), 'blocker_model.pt')))
    else:
        blocker.load_state_dict(torch.load(os.path.join('models', hp.dataset +'_CLER_{}'.format(hp.total_budget) + '_seed_{}'.format(hp.run_id), 'last_blocker_model.pt')), strict=False)
    
    if 'companies' in hp.dataset and load_test:
        embeddingA = get_SentEmb(blocker, datasetA)
        # Save the embeddings to avoid recomputing them again
        torch.save(embeddingA, emb_path)
        return embeddingA, embeddingA
    
    else:
        embeddingA = get_SentEmb(blocker, datasetA)
        embeddingB = get_SentEmb(blocker, datasetB)
        return embeddingA, embeddingB

def load_matcher(hp):
    model_path = os.path.join('models', hp.dataset +'_CLER_{}'.format(hp.total_budget) + '_seed_{}'.format(hp.run_id))
    model = DittoConcatModel(lm = hp.lm)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    if hp.ckpt_type == 'best':
        model.load_state_dict(torch.load(os.path.join(model_path, 'matcher_model.pt')))
    else:
        model.load_state_dict(torch.load(os.path.join(model_path, 'last_matcher_model.pt')), strict=False)
    return model

def gen_testset(topkA, distA, test_idxs, start, end):
    test_data = []
    for idxA in test_idxs:
        for idxB, dist in zip(topkA[idxA][start:end], distA[idxA][start:end]):
            if idxA != idxB:
                test_data.append([idxA, idxB, dist])
    test_data = np.array(test_data)
    return test_data

def pred(model, dataset, batch_size=128):
    iterator = DataLoader(dataset=dataset, batch_size= batch_size, collate_fn=dataset.pad)
    model.eval()
    y_truth, y_pre, y_scores = [], [], []
    e1_list, e2_list = [], []
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        x, _, y = batch   
        x = x.cuda()
        logits = model(x)
        scores = logits.argmax(-1)
        for item in scores.cpu().numpy().tolist():
            y_pre.append(item)
        y_scores.extend(logits.softmax(-1)[:,1].cpu().detach().numpy().tolist())
        del logits
        del scores
    return np.array(y_pre), np.array(y_scores)

def pred_batch(batch):
    x, _, y = batch
    x = x.cuda()
    logits = model(x)
    scores = logits.argmax(-1)
    y_pre = scores.cpu().numpy().tolist()
    y_scores = logits.softmax(-1)[:,1].cpu().detach().numpy().tolist()
    return np.array(y_pre), np.array(y_scores)

def load_test_gt(hp):
    ''' load gt only for test idxs '''
    gt = load_gt(hp)
    gt = gt.reset_index()
    gt = gt.set_index('ltable_id')
    test_idxs = pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_idxs.csv'), index_col=0)
    test_idxs['flag'] = 1 
    test_idxs = test_idxs.set_index('ltable_id')
    df = test_idxs.join(gt, how = 'inner')
    df = df.fillna(0)
    df = df[['rtable_id', 'label']]
    df = df.reset_index()
    df = df.set_index(['ltable_id', 'rtable_id'])
    return df 

def load_test_gt_pairs(hp):
    ''' Load the gt pairs for the test set from test.csv '''
    dataset_path = hp.path

    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'), index_col=0)
    if 'lid' not in test_df.columns:
        test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test_df = test_df.rename(columns = {'lid': 'ltable_id', 'rid': 'rtable_id'})
    test_df, test_ids_dict = translate_ids(dataset_path, test_df, load_test=True)
    gt = test_df[['ltable_id', 'rtable_id', 'label']]
    gt = gt.set_index(['ltable_id', 'rtable_id'])

    test_idxs = np.concatenate([test_df['ltable_id'].values, test_df['rtable_id'].values])
    # Remove the duplicates in the test idxs
    test_idxs = set(test_idxs)
    test_idxs = np.array(list(test_idxs))
    return gt, test_idxs, test_ids_dict

def valid_gap(valid_df, p):
    def cal_gap(df_sub):
        pos = df_sub[df_sub.label==1]
        if len(pos) == 0:
            return -1
        pos_sim = np.min(pos['sim'].values)
        neg = df_sub[df_sub.label==0]['sim'].values
        if len(neg) == 0:
            return -1
        neg_sim = np.max(neg)
        return pos_sim - neg_sim
    gaps = valid_df.groupby('ltable_id').apply(cal_gap).values
    return np.percentile(gaps[gaps>0], p)

def update(df_all, k, test_idxs, gap, min_sim, thr):
    rm_testids = set()
    for lid in tqdm(test_idxs, desc='Updating test set', total=len(test_idxs)):
        dfsub = df_all[df_all.ltable_id==lid] # [ltable_id, rtable_id, sim, pred]
        dfsub = dfsub.sort_values(by = 'sim', ascending = False)
        sims = dfsub['sim'].values
        if (((dfsub['pred']==1).sum() > 0) and (np.sum(dfsub['pred'].values[-k:]) == 0)) or (((dfsub['pred']==1).sum() == 0) and (len(dfsub)==50 or (sims[-1] < min_sim))):
            rm_testids.add(lid)

    return set(test_idxs) - rm_testids

def load_preds(eval_path, batch_size):
    # Get the last batch number from the last saved file in the eval directory
    eval_files = os.listdir(eval_path)
    eval_files = sorted(eval_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    last_file = eval_files[-1]
    last_batch = int(last_file.split('.')[0].split('_')[-1])
    preds = []
    scores = []
    for i in range(last_batch + 1):
        batch_preds = np.load(os.path.join(eval_path, 'preds_'+str(i)+'.npy'))
        batch_scores = np.load(os.path.join(eval_path, 'scores_'+str(i)+'.npy'))
        preds.append(batch_preds)
        scores.append(batch_scores)
        if i!= last_batch:
            assert len(batch_preds) == batch_size and len(batch_scores) == batch_size 
        
    y_pred = np.concatenate(preds)
    y_scores = np.concatenate(scores)
    return y_pred, y_scores


if __name__=="__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="wdc/shoes")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--add_token", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--CLlogdir", type=str, default="CL-sep-sup_0104")
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--total_budget", type=int, default=500)
    parser.add_argument("--warmup_budget", type=int, default=400)
    parser.add_argument("--active_budget", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--topK", type=int, default=5) # This is the topK used during training, the number of candidates to consider in evaluation can be different
    parser.add_argument("--balance", type=bool, default=False)
    parser.add_argument("--valid_size", type=int, default=200)
    parser.add_argument("--blocker_type", type=str, default='sentbert') # sentbert/magellan
    parser.add_argument("--validation_with_pseudo", type=bool, default=False)
    parser.add_argument("--aug_type", type=str, default='random')
    parser.add_argument("--num_iter", type=int, default=5)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--ckpt_type", type=str, default='last')
    
    hp = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu

    alpha = 1.65
    dataset = hp.dataset
    if 'wdc' in dataset:
        hp.path = "data/raw/wdc80_pair"
    elif 'companies' in dataset:
        hp.path = "data/raw/synthetic_data/seed_0/companies"
    elif 'camera' in dataset:
        hp.path = 'data/raw/camera'
    elif 'monitor' in dataset:
        hp.path = 'data/raw/monitor'
    elif 'musicbrainz' in dataset:
        hp.path = 'data/raw/musicbrainz'
    else:
        hp.path = "data/ER-Magellan"
        if 'Abt' in dataset:
            hp.dataset = os.path.join("Textual", dataset)
        else:
            hp.dataset = os.path.join("Structured", dataset)    

    if 'companies' in dataset or 'wdc' in dataset or 'camera' in dataset or 'monitor' in dataset or 'musicbrainz' in dataset:
        gt, test_idxs, test_ids_dict = load_test_gt_pairs(hp)
    else:
        gt = load_test_gt(hp)
        test_idxs = set(list(pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_idxs.csv'), index_col=0)['ltable_id'].values))
    
    attr_listA, entity_listA, attr_listB, entity_listB = load_attributes(hp.path, load_test=True)
    model = load_matcher(hp)

    all_BK_rec_list, all_f1_list, all_pre_list, all_rec_list = [], [], [], []
    embeddingA, embeddingB = get_emb(hp, load_test=True)
    topkA, distA, sim_score = get_topK(embeddingA, embeddingB, topK = min(5, len(embeddingB)), hp=hp)
    start, end, k = 0, 5, 5
    if 'companies' in dataset or 'wdc' in dataset or 'camera' in dataset or 'monitor' in dataset or 'musicbrainz' in dataset:
        valid_df = load_valid_df(hp)
        _, entity_listA_val, _,  entity_listB_val = load_attributes(hp.path)
        valid_set = GTDatasetWithLabel(valid_df.values, entity_listA_val, entity_listB_val, attr_listA, lm=hp.lm, concat=True, shuffle=False)
    else:
        valid_df = pd.read_csv(os.path.join(hp.path, hp.dataset, 'valid0207.csv'))
        valid_set = GTDatasetWithLabel(valid_df.values, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
    
    y_pred, y_scores = pred(model, valid_set, batch_size=128)
    valid_df['score'] = y_scores
    thr = np.percentile(valid_df[(valid_df.label==0)&(valid_df.score<0.5)]['score'].values, 100-hp.p)
    print('neg thr', thr)
    print('pos thr',valid_df[(valid_df.label==0)]['score'].min())

    if 'companies' in dataset or 'wdc' in dataset or 'camera' in dataset or 'monitor' in dataset or 'musicbrainz' in dataset:
        # We need to compute the topK for the validation set
        embeddingA_val, embeddingB_val = get_emb(hp)
        topkA_val, distA_val, sim_score_val = get_topK(embeddingA_val, embeddingB_val, topK = min(5, len(embeddingB_val)))
        sim = []
        for l, r, _ in valid_df[['ltable_id', 'rtable_id', 'label']].values:
            sim.append(sim_score_val[l][r])
    else:
        sim = []
        for l, r, _ in valid_df[['ltable_id', 'rtable_id', 'label']].values:
            sim.append(sim_score[l][r])

    valid_df['sim'] = sim
    gap = valid_gap(valid_df, hp.p)
    pos_sim = valid_df[valid_df.label==1]['sim'].values
    min_sim = np.min(pos_sim)
    print('gap', gap, 'min sim', min_sim, 'std sim', np.std(pos_sim))

    df_all = None
    iter = 0
    batch_size = 128
    while len(test_idxs) > 0:
        iter += 1
        print(iter, len(test_idxs))
        test_data = gen_testset(topkA, distA, test_idxs, start, end)
        test_set = GTDatasetWithLabel(test_data, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
        model.eval()

        if 'companies' in dataset:
            # CLER produces 700k candidate pairs, so we need to evaluate the test set in batches
            eval_path = os.path.join('models', hp.dataset + '_CLER_' + str(hp.total_budget) + '_seed_' + str(hp.run_id), 'eval_iter_{}_batch_size_{}'.format(iter,batch_size))
            if not os.path.exists(eval_path):
                os.makedirs(eval_path)
            # Check first if the evaluation has been partially or fully done
            # Get all the file names in the eval directory
            eval_files = os.listdir(eval_path)
            if len(eval_files) > 0:
                # Order the files by the batch number
                eval_files = sorted(eval_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
                last_file = eval_files[-1]
                # Get the last batch number
                batch_num = int(last_file.split('.')[0].split('_')[-1])
                if batch_num == len(test_data)//batch_size:
                    # The evaluation has been fully done
                    y_pred, y_scores = load_preds(eval_path, batch_size)
                else:
                    # The evaluation has been partially done
                    # Evaluate the remaining batches
                    test_set = GTDatasetWithLabel(test_data[batch_size*(batch_num+1):], entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
                    iterator = DataLoader(dataset= test_set, batch_size= batch_size, collate_fn=test_set.pad)

                    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
                        preds, scores = pred_batch(batch)
                        np.save(os.path.join(eval_path, 'preds_'+str(batch_num + i + 1)+'.npy'), preds)
                        np.save(os.path.join(eval_path, 'scores_'+str(batch_num + i + 1)+'.npy'), scores)

                    y_pred, y_scores = load_preds(eval_path, batch_size)
                        

            else:
                # No evaluation has been done yet
                preds = []
                scores = []
                iterator = DataLoader(dataset= test_set, batch_size= batch_size, collate_fn=test_set.pad)

                for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
                    preds, scores = pred_batch(batch)
                    np.save(os.path.join(eval_path, 'preds_'+str(i)+'.npy'), preds)
                    np.save(os.path.join(eval_path, 'scores_'+str(i)+'.npy'), scores)

                y_pred, y_scores = load_preds(eval_path, batch_size)
        else:
            y_pred, y_scores = pred(model, test_set, batch_size=128)

        df_test = pd.DataFrame({'ltable_id': test_data[:,0], 'rtable_id': test_data[:,1], 'sim': test_data[:,2], 'pred': y_pred, 'score': y_scores})
        if df_all is None:
            df_all = df_test
        else:
            df_all = pd.concat([df_all, df_test])
        test_idxs = update(df_all, k, test_idxs, gap, min_sim-alpha*np.std(pos_sim), thr)              
        start = end
        end += k
        if start >= min(len(topkA[0]), 50):
            break
    
    df_all = df_all.set_index(['ltable_id', 'rtable_id'])
    df1 = gt.join(df_all)
    BK_recall = len(df1[(df1.label==1) & (pd.isna(df1.pred)==False)]) / float(len(df1[(df1.label==1)]))
    # matcher precision recall
    df2 = df_all.join(gt)
    df2 = df2.fillna(0)
    pre = precision_score(df2['label'].values, df2['pred'].values)
    rec = recall_score(df2['label'].values, df2['pred'].values)
    rec = BK_recall * rec 
    try:
        f1 = (2 * pre * rec) / (pre + rec)
    except:
        f1 = 0

if 'companies' in hp.dataset:
    # We need to save the results in the companies dataset after converting the ids back to the original ids
    df_all = translate_ids_back(df_all, test_ids_dict)
    df_all = df_all.rename(columns = {'ltable_id': 'lid', 'rtable_id': 'rid', 'pred': 'prob'})
    df_all = df_all.drop(columns = ['sim' , 'score'])
    df_all = df_all[['lid', 'rid', 'prob']]
    results_path = os.path.join('data', 'results', 'synthetic_companies', hp.dataset + '_CLER_' + str(hp.total_budget) + '_seed_' + str(hp.run_id))
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    df_all.to_csv(os.path.join(results_path, 'pairwise_matches_preds.csv'), index=False)
elif 'wdc' in hp.dataset or 'camera' in hp.dataset or 'monitor' in hp.dataset or 'musicbrainz' in hp.dataset:
    df_all = translate_ids_back(df_all, test_ids_dict)
    df_all = df_all.rename(columns = {'ltable_id': 'lid', 'rtable_id': 'rid', 'pred': 'prob'})
    df_all = df_all.drop(columns = ['sim' , 'score'])
    df_all = df_all[['lid', 'rid', 'prob']]
    results_path = os.path.join('data', 'results', hp.dataset, hp.dataset + '_CLER_' + str(hp.total_budget) + '_seed_' + str(hp.run_id))
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    df_all.to_csv(os.path.join(results_path, 'pairwise_matches_preds.csv'), index=False)


print('run_id_'+str(hp.run_id), hp.topK, 'Final BK size', hp.total_budget, len(df_all))
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test BK Recall', hp.total_budget, BK_recall)
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test F1', hp.total_budget, f1)
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test Precision', hp.total_budget, pre)
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test Recall', hp.total_budget, rec)

final_time = time.time() - start_time
hours, rem = divmod(final_time, 3600)
minutes, seconds = divmod(rem, 60)
print("Total testing time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
# Save the training time to the model folder
with open(os.path.join('models', hp.dataset + '_CLER_' + str(hp.total_budget) + '_seed_' + str(hp.run_id), 'elapsed_time_test_dynamic.txt'), 'w') as f:
    f.write("Total testing time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print("Testing Done!")