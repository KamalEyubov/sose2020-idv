from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, vote_pred, votenum, vote_score, mode):
        self.vote_pred = vote_pred
        self.votenum = votenum
        self.vote_score = vote_score
        self.vote_pred[vote_pred <= (votenum/2)] = 0
        self.vote_pred[vote_pred > (votenum/2)] = 1
        self.vote_score = vote_score/votenum

        self.targetlist = []
        self.predlist = []

        self.TP = 0
        self.TN = 0
        self.FN = 0
        self.FP = 0

        self.precisionHistory = []
        self.recallHistory = []
        self.f1History = []
        self.aucHistory = []
        self.accHistory = []

        self.mode = mode

    def update(self,predlist, targetlist, scorelist):
        self.vote_pred += predlist
        self.vote_score += scorelist
        self.targetlist = targetlist
        self.predlist = predlist

    def computeStatistics(self):
        self.vote_pred[self.vote_pred <= (self.votenum/2)] = 0
        self.vote_pred[self.vote_pred > (self.votenum/2)] = 1
        self.vote_score = self.vote_score/self.votenum

        self.TP = ((self.vote_pred == 1) & (self.targetlist == 1)).sum()
        self.TN = ((self.vote_pred == 0) & (self.targetlist == 0)).sum()
        self.FN = ((self.vote_pred == 0) & (self.targetlist == 1)).sum()
        self.FP = ((self.vote_pred == 1) & (self.targetlist == 0)).sum()

        self.precisionHistory.append(self.TP / (self.TP + self.FP))
        self.recallHistory.append(self.TP / (self.TP + self.FN))
        self.accHistory.append((self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN))
        self.aucHistory.append(roc_auc_score(self.targetlist, self.vote_score))
        r = self.getRecall()
        p = self.getPrecision()
        self.f1History.append(2 * r * p / (r + p))

        self.vote_pred = np.zeros(self.vote_pred.__len__())
        self.vote_score = np.zeros(self.vote_score.__len__())


    def getRecall(self):
        return np.mean(self.recallHistory)

    def getPrecision(self):
        return np.mean(self.precisionHistory)

    def getF1(self):
        return np.mean(self.f1History)

    def getAccuracy(self):
        return np.mean(self.accHistory)

    def getAUC(self):
        return np.mean(self.aucHistory)

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
        tickmarks = np.arange(confusion.shape[0])
        fig, ax = plt.subplots()
        pos = plt.imshow(confusion)
        fig.colorbar(pos)
        plt.yticks(tickmarks,['Non_Covid', 'Covid'])
        plt.xticks(tickmarks,['Non_Covid', 'Covid'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_'+self.mode+'.png')
