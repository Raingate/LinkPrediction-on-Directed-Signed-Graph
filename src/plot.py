import matplotlib.pyplot as plt
import numpy as np

# from main_classify_operations import embedding

AUCs,F1_scores,Acc_pos,Acc_neg = np.load("results10.npy")
# print(AUCs,F1_scores,Acc_pos,Acc_neg)
# print(AUCs)
# print(F1_scores)
# plt.plot(AUCs)
# plt.show()

# experiments 

# training-set size for node2vec
# [0.7000895, 0.709007, 0.7122675, 0.720296, 0.7191335, 0.72537] [0.57, 0.571, 0.5945, 0.601, 0.6415, 0.6605]

# training-set size for v0
# [0.556, 0.607,0.616,0.647,0.647 ,0.643]    [0.540, 0.567, 0.576, 0.597, 0.599 ,0.59]

# training-set size for our-best
# [0.67741, 0.7542605,0.799823,0.8279,0.838614 ,0.8626445]    [0.631, 0.6885, 0.7275, 0.7515, 0.764 ,0.7865]

# embedding size for our-best
# [0.74146,  0.7955325, 0.807645, 0.8455125, 0.8626445]   [0.6795, 0.7255, 0.738, 0.773, 0.7865]

# sampleLen for out-best
# [0.8489, 0.8589, 0.8626445 ,0.8377, 0.8466 ]   [0.783,0.807, 0.7865 ,0.786, 0.776 ]

# choice of classifier
# average: 0.7708 0.7065
# hadma: 0.6321 0.5765
# 

train_set_size = [0.1,0.2,0.4,0.6,0.8,1.0]
auc1,fscore1 = [0.7090895, 0.700007, 0.7202675, 0.712296, 0.7191335, 0.72537], [0.57, 0.571, 0.5945, 0.601, 0.6415, 0.6605]
auc2,fscore2 = [0.556, 0.607,0.616,0.647,0.647 ,0.643]  ,  [0.540, 0.567, 0.576, 0.597, 0.599 ,0.59]
auc3,fscore3 = [0.67741, 0.7542605,0.799823,0.8279,0.838614 ,0.8626445]  ,  [0.631, 0.6885, 0.7275, 0.7515, 0.764 ,0.7865]

plt.plot(train_set_size,auc1,'-xr',label="node2vec")
plt.plot(train_set_size,auc2,'-og',label="coupled")
plt.plot(train_set_size,auc3,'-^b',label="decoupled")
plt.xlabel("training set size")
plt.ylabel("AUC")
plt.xlim((0.09,1.01))
plt.ylim(0.5,0.9)
plt.legend()
plt.show()

plt.plot(train_set_size,fscore1,'-xr',label="node2vec")
plt.plot(train_set_size,fscore2,'-og',label="coupled")
plt.plot(train_set_size,fscore3,'-^b',label="decoupled")
plt.xlabel("training set size")
plt.ylabel("Micro f1 score")
plt.xlim((0.09,1.01))
plt.ylim(0.5,0.8)
plt.legend()
plt.show()

plt.plot(AUCs,'-xr',label="AUC")
plt.plot(F1_scores,'-^b',label="f1-score")
plt.xlabel("training epochs")
plt.ylabel("metrics")
# plt.xlim((0.09,1.01))
plt.ylim(0.5,0.9)
plt.legend()
plt.show()


aucs,f1scores = [0.74146,  0.7955325, 0.807645, 0.8455125, 0.8626445] ,  [0.6795, 0.7255, 0.738, 0.773, 0.7865]
embedding_size = [16,32,64,128,256]
plt.plot(embedding_size,aucs,'-xr',label="AUC")
plt.plot(embedding_size,f1scores,'-^b',label="f1-score")
plt.xlabel("training epochs")
plt.ylabel("metrics")
# plt.xlim((0.09,1.01))
plt.ylim(0.5,0.9)
plt.legend()
plt.show()