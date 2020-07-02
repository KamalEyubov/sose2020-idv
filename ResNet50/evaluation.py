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

        self.precision = 0
        self.recall = 0
        # self.F1 = []
        # self.AUC = []
        # self.accuracy = []

        self.mode = mode

    def update(self,predlist, targetlist, scorelist):
        self.vote_pred += predlist
        self.vote_score += predlist
        self.vote_pred[self.vote_pred <= (self.votenum/2)] = 0
        self.vote_pred[self.vote_pred > (self.votenum/2)] = 1
        self.vote_score = self.vote_score/self.votenum

        self.targetlist = targetlist
        self.predlist = predlist

        self.TP = ((self.vote_pred == 1) & (self.targetlist == 1)).sum()
        self.TN = ((self.vote_pred == 0) & (self.targetlist == 0)).sum()
        self.FN = ((self.vote_pred == 0) & (self.targetlist == 1)).sum()
        self.FP = ((self.vote_pred == 1) & (self.targetlist == 0)).sum()

        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)


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
        tickmarks = np.arange(confusion.shape[0])
        fig = plt.subplots()
        plt.imshow(confusion)
        plt.yticks(tickmarks,['Non_Covid', 'Covid'])
        plt.xticks(tickmarks,['Non_Covid', 'Covid'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_'+self.mode+'.png')
