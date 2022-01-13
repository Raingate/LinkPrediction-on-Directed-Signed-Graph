import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
import collections
import csv
import random
from sklearn.metrics import f1_score
from sklearn import metrics


train_path = '../data/parsed.csv'
path_emb_src = '../model/emb_src.pth'
path_emb_tgt = '../model/emb_tgt.pth'
path_C = '../model/c.pth'
path_classifier = '../model/classifier.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_model = True
num_edges = 198275
num_nodes = 11381+2
num_epoch_1 = 20
num_epoch_2 = 30
pos_edge = 141521
neg_edge = 40363
test_num = 1000
len_sample = 10

def create_graphs(csv_path,training_ratio=1):
    '''
    input:
        csv_path: path to the csv data
        training_ratio: the ratio of used training dataset

    return:
        Graph_p: dict of dict, positive edges for embedding training
        Graph_n: dict of dict, negtive edges for embedding training
        Graph_all: dict of dict, all edges including training and testing data
        test_set_p: list of tuple (int(vj), int(vi), int(w)), used for final metrics
        test_set_n: list of tuple (int(vj), int(vi), int(w)), used for final metrics
        train_cls_p: list of tuple (int(vj), int(vi), int(w)), used for classification training
        train_cls_n: list of tuple (int(vj), int(vi), int(w)), used for classification training
    '''
    sample_interval = round(training_ratio*10)
    Graph_p = collections.defaultdict(dict)
    Graph_n = collections.defaultdict(dict)
    Graph_all = collections.defaultdict(dict)
    test_set_p = []
    test_set_n = []
    train_cls_p = []
    train_cls_n = []

    edges = []
    pos_edges = []
    neg_edges = []
    with open (csv_path) as f1:
        reader = csv.reader(f1)
        # header = next(reader)
        global num_edges
        for i in range(num_edges):
            vj, vi, w = next(reader)
            vj, vi, w = int(vj), int(vi), int(w)
            if abs(int(w)) == 1:
                edges.append((vj, vi, w))
            if int(w) == 1:
                pos_edges.append((vj, vi, w))
            if int(w) == -1:
                neg_edges.append((vj, vi, w))
        print("edges num:", len(edges),len(pos_edges),len(neg_edges))
    f1.close()

    random.shuffle(edges)
    for i in range(len(edges)):
        vj, vi, w = edges[i]
        Graph_all[int(vi)][int(vj)] = int(w)
        if w>0:
            if len(test_set_p)<test_num:
                test_set_p.append((int(vj), int(vi), int(w)))
                continue
            if i%10 < sample_interval:
                Graph_p[int(vi)][int(vj)] = 1
                train_cls_p.append((int(vj), int(vi), int(w)))

        if w<0:
            if len(test_set_n)<test_num:
                test_set_n.append((int(vj), int(vi), int(w)))
                continue
            if i%10 < sample_interval:
                Graph_n[int(vi)][int(vj)] = -1
                train_cls_n.append((int(vj), int(vi), int(w)))

    return Graph_p, Graph_n, Graph_all, test_set_p, test_set_n, train_cls_p, train_cls_n

class embedding(nn.Module):
    def __init__(self, graph_size,h_size):
        super(embedding, self).__init__()
        self.graph_size = graph_size
        self.embedding = nn.Embedding(self.graph_size,h_size)
        print("h_size:", h_size)
    
    def forward(self,x):
        y = self.embedding(x)
        return y

class C(nn.Module):
    def __init__(self,h_size):
        super(C, self).__init__()
        self.positive_C = nn.Linear(h_size,h_size)
        self.negtive_C = nn.Linear(h_size,h_size)
        self.size = h_size

    def forward(self,x,positive_num):
        """
        x: 2d tensor , shape= (positive_num+ negtive_num, size)
        return: 1d tensor, shape = (1,size)
        """
        positive_sum = torch.mean(x[:positive_num],dim=0)
        negtive_sum = torch.mean(x[positive_num:],dim=0)
        return_vec = self.positive_C(positive_sum) + self.positive_C(negtive_sum)
        return return_vec.reshape(1,-1)

class classifier(nn.Module):
    def __init__(self,input_size ):
        super(classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Tanh()
        )

    def forward(self,x):
        return self.layers(x)

def random_walk_sample(graph_p,graph_n,graph_all, sample_len):
    '''
    input:
        graph_p: dict of dict, graph of positive edges
        graph_n: dict of dict, graph of negative edges
        graph_all: dict of dict, graph of all edges
        sample_len: num of positive or negative neighbors sampled for every node

    returns:
        train_set: list of tuple (sample_sequence,fake_edge,num_pos)
            sample_sequence: a list of sampled neighbors(int)
            fake_edge: Boolean, whether the node use fake edge if it does not have positive or negative neighbors
            num_pos: num of sampled positive edges
    '''
    train_set = []
    start_points = list( graph_all.keys() )
    random.shuffle(start_points)

    for i in range(len(start_points)):
        sample_sequence = []
        v = start_points[i]
        sample_sequence.append(v)
        fake_edge = False
        num_pos = 0
        
        pos_neibors = list( graph_p[v].keys() )
        random.shuffle(pos_neibors)
        for j in range(sample_len):
            
            # print(len(pos_neibors))
            if len(pos_neibors)==0:
                sample_sequence.append(num_nodes-2)
                fake_edge = True
                break
            else:
                u = pos_neibors.pop(0)
                sample_sequence.append(u)
                if len(pos_neibors)==0:
                    break

        num_pos = len(sample_sequence)-1

        neg_neibors = list( graph_n[v].keys() )
        random.shuffle(neg_neibors)
        for j in range(sample_len):
            if len(neg_neibors)==0:
                # print("$$$$$$$")
                sample_sequence.append(num_nodes-1)
                fake_edge = True
                break
            else:
                u = neg_neibors.pop(0)
                sample_sequence.append(u)
                if len(neg_neibors)==0:
                    break

        train_set.append((sample_sequence,fake_edge,num_pos))
        
    return train_set #1+2*sample_len

def main(h_size=256,training_ratio = 1,len_sample = 10):
    # global h_size
    GraphP, GraphN, Graph_all, Test_set_p, Test_set_n, train_clsp, train_clsn = create_graphs(train_path,training_ratio)    # read csv and create graph as a 'defaultdict'
    # training Positive
    Net_embedding_source = embedding(num_nodes,h_size=h_size).to(device)
    Net_embedding_target = embedding(num_nodes,h_size=h_size).to(device)
    Net_C = C(h_size=h_size).to(device)
    Net_classifier = classifier(input_size=2*h_size).to(device)

    # load model if possible, default to retrain
    try:
        weight_Net_embedding_source = torch.load(path_emb_src)
        Net_embedding_source.load_state_dict(weight_Net_embedding_source)

        weight_Net_embedding_target = torch.load(path_emb_tgt)
        Net_embedding_target.load_state_dict(weight_Net_embedding_target)

        weight_Net_C = torch.load(path_C)
        Net_C.load_state_dict(weight_Net_C)

        weight_Net_classifier = torch.load(path_classifier)
        Net_classifier.load_state_dict(weight_Net_classifier)
    except:
        print("no avaliable model!")


    optimizer_emb_src = optim.Adam(Net_embedding_source.parameters(), lr = 1e-2)
    optimizer_emb_tgt = optim.Adam(Net_embedding_target.parameters(), lr = 1e-2)
    optimizer_C = optim.Adam(Net_C.parameters(), lr = 1e-3)
    optimizer_classifier = optim.Adam(Net_classifier.parameters(), lr = 1e-3)
    criterion = nn.MSELoss()

    # train the embedding
    for epoch in range(num_epoch_1):
        print("entered epoch",epoch+1,'...')

        # construct train samples using random walk on Graph
        train_set = random_walk_sample(GraphP,GraphN, Graph_all, len_sample)

        # no minibatch
        total_loss1 = []
        total_loss2 = []
        p_score = []
        n_score = []
        fake_num = 0
        bar = tqdm(range(len(train_set)))# #1000
        for i in bar:
            train_data = train_set[i][0]
            fake_edge = train_set[i][1]
            num_pos = train_set[i][2]
            train_data = torch.LongTensor(train_data).to(device)
            embedded_neibors = Net_embedding_source(train_data[1:]) #(2*sample_len)*size
            embedded_tmp_node = Net_embedding_target(train_data[:1]) #1*size
            weighted_neibors = Net_C(embedded_neibors,num_pos) # shape = (1,size)
            similarity = torch.cosine_similarity(embedded_tmp_node,weighted_neibors)
            # loss1 = - torch.log(torch.sigmoid(similarity))
            loss1 = - torch.log((similarity+1.0)/2.0)
            total_loss1.append(loss1.detach().cpu().numpy())



            optimizer_C.zero_grad()
            optimizer_emb_src.zero_grad()
            optimizer_emb_tgt.zero_grad()
            loss1.backward()

            optimizer_C.step()
            optimizer_emb_src.step()
            optimizer_emb_tgt.step()
        print("Loss of epoch", epoch + 1, ':', np.mean(total_loss1))

        # save model
        if save_model:
            src_weight = Net_embedding_source.state_dict()
            torch.save(src_weight, path_emb_src)
            tgt_weight = Net_embedding_source.state_dict()
            torch.save(tgt_weight, path_emb_tgt)
            c_weight = Net_C.state_dict()
            torch.save(c_weight,path_C)

    # train the classifier
    best_AUC = 0.0
    AUCs = []
    F1_scores = []
    Acc_pos = []
    Acc_neg = []
    for epoch in range(num_epoch_2):
        print("entered epoch", epoch + 1, '...')
        random.shuffle(train_clsp)
        random.shuffle(train_clsn)
        print(len(train_clsp),len(train_clsn))
        train_set_cls = train_clsp[:int(20000*training_ratio)]
        nnn = train_clsn[:int(20000*training_ratio)]
        train_set_cls.extend(nnn)
        random.shuffle(train_set_cls)

        for i in tqdm(range(len(train_set_cls))):
            src = torch.LongTensor([train_set_cls[i][0]]).to(device)
            emb_src = Net_embedding_source(src)
            tgt = torch.LongTensor([train_set_cls[i][1]]).to(device)
            emb_tgt = Net_embedding_source(tgt)
            emb_cat = torch.cat((emb_src, emb_tgt), dim=1)
            classify_result = Net_classifier(emb_cat).reshape(-1)
            loss2 = criterion(classify_result, torch.tensor([train_set_cls[i][2]],dtype=torch.float32).to(device).reshape(-1))

            optimizer_classifier.zero_grad()
            loss2.backward()
            optimizer_classifier.step()


        # compute metrics on test-set
        count = 0.0
        p_pool = []
        n_pool = []
        p_count = 0
        n_count = 0
        pred_label = []
        true_label = []
        for i in range(test_num):
            src = torch.LongTensor([Test_set_p[i][0]]).to(device)
            emb_src = Net_embedding_source(src)
            tgt = torch.LongTensor([Test_set_p[i][1]]).to(device)
            emb_tgt = Net_embedding_source(tgt)
            emb_cat = torch.cat((emb_src, emb_tgt), dim=1)
            classify_result_p = Net_classifier(emb_cat).detach().cpu().numpy().reshape(-1)
            p_pool.append(classify_result_p)
            true_label.append(1)
            p_count += classify_result_p / test_num

        for j in range(test_num):
            src = torch.LongTensor([Test_set_n[j][0]]).to(device)
            emb_src = Net_embedding_source(src)
            tgt = torch.LongTensor([Test_set_n[j][1]]).to(device)
            emb_tgt = Net_embedding_source(tgt)
            emb_cat = torch.cat((emb_src, emb_tgt), dim=1)
            classify_result_n = Net_classifier(emb_cat).detach().cpu().numpy().reshape(-1)
            n_pool.append(classify_result_n)
            true_label.append(0)
            n_count += classify_result_n / test_num

        pred_score = p_pool.copy()
        pred_score.extend(n_pool)
        pred_score = list(np.array(pred_score).reshape(-1))
        pred_score_normalized = list(((np.array(pred_score,dtype=np.float64)+1.0)/2.0).reshape(-1))
        fpr, tpr, thresholds = metrics.roc_curve(true_label, pred_score_normalized)
        auc = metrics.auc(fpr, tpr)

        acc_p = 0.0
        acc_n = 0.0
        for i in range(len(pred_score)):
            if pred_score[i]>0:
                pred_label.append(1)
            else:
                pred_label.append(0)
            if pred_score[i]>0 and true_label[i]==1:
                acc_p += 1.0/test_num
            if pred_score[i]<=0 and true_label[i]==0:
                acc_n += 1.0/test_num
        
        f1score = f1_score(true_label,pred_label,average="micro")

        print("AUC:",round(auc*100,3),"%")
        print("micro f1 score: ",round(f1score*100,3),"%")
        AUCs.append(auc)
        F1_scores.append(f1score)
        Acc_pos.append(acc_p)
        Acc_neg.append(acc_n)

        if auc > best_AUC:
            if save_model:
                weight_classifier = Net_classifier.state_dict()
                torch.save(weight_classifier,path_classifier)
                print("save classifier!")
            best_AUC = auc
        print("")
        

    return AUCs,F1_scores,Acc_pos,Acc_neg

if __name__ == '__main__':
    AUCs,F1_scores,Acc_pos,Acc_neg = main()
    print(AUCs,F1_scores,Acc_pos,Acc_neg)