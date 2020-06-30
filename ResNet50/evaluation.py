from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, vote_pred, votenum, vote_score, targetlist, predlist, valset, mode):
        self.vote_pred = vote_pred
        self.vote_score = vote_score
        self.vote_pred[vote_pred <= (votenum/2)] = 0
        self.vote_pred[vote_pred > (votenum/2)] = 1
        self.vote_score = vote_score/votenum

        self.targetlist = targetlist
        self.predlist = predlist
        # print('vote_pred', vote_pred)
        # print('targetlist', targetlist)
        self.TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        self.TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        self.FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        self.FP = ((vote_pred == 1) & (targetlist == 0)).sum()

        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)

        self.mode = mode

    def getF1(self):
        return 2 * self.recall * self.precision / (self.recall + self.precision)

    def getAccuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def getAUC(self):
        return roc_auc_score(self.targetlist, self.vote_score)

    def getConfusion(self):
        confusion = confusion_matrix(self.targetlist, self.predlist)
        return confusion/np.linalg.norm(confusion)


    def plotEval(self):
        F1 = self.getF1()
        accuracy = self.getAccuracy()
        AUC = self.getAUC()
        x = np.arange(3)
        fig = plt.subplots()
        plt.bar(x, [F1, accuracy, AUC])
        plt.xticks(x, ('F1', 'Accuracy', 'AUC'))
        plt.savefig('evaluation_'+self.mode+'.png')

    def plotConfusion(self):
        confusion = self.getConfusion()
        print(confusion)
        tickmarks = np.arange(confusion.shape[0])
        fig = plt.subplots()
        plt.imshow(confusion)
        plt.yticks(tickmarks,['Non_Covid', 'Covid'])
        plt.xticks(tickmarks,['Non_Covid', 'Covid'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_'+self.mode+'.png')





# print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
# print('TP+FP',TP+FP)
# p = TP / (TP + FP)
# print('precision',p)
# p = TP / (TP + FP)
# r = TP / (TP + FN)
# print('recall',r)
# F1 = 2 * r * p / (r + p)
# acc = (TP + TN) / (TP + TN + FP + FN)
# print('F1',F1)
# print('acc',acc)
# AUC = roc_auc_score(targetlist, vote_score)
# print('AUCp', roc_auc_score(targetlist, vote_pred))
# print('AUC', AUC)
