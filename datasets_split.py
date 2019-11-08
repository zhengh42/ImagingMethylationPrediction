import numpy as np
import pandas as pd


class DatasetSplit(object):

    def __init__(self, config, dataset, iter):

        self.config = config
        self.dataset = dataset
        self.iter = iter
        self.data_split()

    def data_split(self):

        self.tasks = list(self.dataset.DMvalues.columns)
        indices = np.random.permutation(self.dataset.morpho_context.index)
        self.dataset.morpho_context = self.dataset.morpho_context.loc[indices, :]
        self.dataset.DMvalues = self.dataset.DMvalues.loc[indices, :]
        idx_test = int((1-self.config.test_size)*len(self.dataset.DMvalues))
        self.X_train, self.X_test = np.split(self.dataset.morpho_context, [idx_test])
        print('Number of patients', self.dataset.DMvalues.shape[0])
        print('Number of genes', self.dataset.DMvalues.shape[1])
        self.y_train, self.y_test = np.split(self.dataset.DMvalues, [idx_test])
        pd.DataFrame(self.X_train).to_csv("output/%s/%s/X_train_%s.txt" % (self.config.source, self.config.folder, self.iter))
        pd.DataFrame(self.y_train).to_csv("output/%s/%s/y_train_%s.txt" % (self.config.source, self.config.folder, self.iter))
        pd.DataFrame(self.X_test).to_csv("output/%s/%s/X_test_%s.txt" % (self.config.source, self.config.folder,  self.iter))
        pd.DataFrame(self.y_test).to_csv("output/%s/%s/y_test_%s.txt" % (self.config.source, self.config.folder,  self.iter))
