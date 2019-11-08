import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 1147
class Config(object):

    def __init__(self, source="GBM", folder="test", models="LogisticRegression", classifier="rf", test_size=0.25, clustering="hierarchical", n_clusters=5, n_tasks=1147):

        # Input/Output
        self.source = source
        self.folder = folder

        # experiment -> label
        self.models = models
        self.classifier = classifier

        # Data parameters
        self.test_size = test_size
        self.clustering = clustering
        self.n_clusters = n_clusters
        self.n_tasks = n_tasks

        # Saving parameters
        if os.path.isdir("output/%s/%s/%s" % (self.source, self.folder, self.models)):
            shutil.rmtree("output/%s/%s/%s" % (self.source, self.folder, self.models))
        os.makedirs("output/%s/%s/%s" % (self.source, self.folder, self.models))

        # Figure to save all ROC and PR curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_xlabel("False Positive Rate", size=14)
        ax1.set_ylabel("True Positive Rate", size=14)
        ax1.set_title("Average ROC curves")
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])
        ax2.set_xlabel("Recall", size=14)
        ax2.set_ylabel("Precision", size=14)
        ax2.set_title("Average PR curves")
        self.fig, self.ax1, self.ax2 = fig, ax1, ax2
