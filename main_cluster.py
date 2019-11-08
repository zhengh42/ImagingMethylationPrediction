import numpy as np
import pandas as pd
from config import Config
from models import ModelML
from datasets import DatasetML
from datasets_split import DatasetSplit
import matplotlib.pyplot as plt


for clusternumber in range(3, 8):
    plt.close()
    #config = Config(folder="HC_%s_clusters" % (clusternumber), source="RCC", models=None, clustering="hierarchical", n_clusters=clusternumber, n_tasks=1147)
    config = Config(folder="HC_%s_clusters" % (clusternumber), source="glioma", models=None, clustering="hierarchical", n_clusters=clusternumber, n_tasks=1147)
    dataset = DatasetML(config)

    pd.DataFrame(dataset.DMvalues).to_csv("output/%s/%s/dataset.DMvalues.txt" % (config.source, config.folder))

    values = ["lr", "rf", "svm", "nb", "nn", "ada"]
    models = ["LogisticRegression", "RandomForest", "SupportVectorMachines", "NaiveBayes", "NeuralNetwork", "AdaBoost"]

    for iter in range(30):
        DatasetSplit(config, dataset, iter)
        X_train = pd.read_csv("output/%s/%s/X_train_%s.txt" % (config.source, config.folder, iter), index_col=0, header=0, sep=",")
        X_test = pd.read_csv("output/%s/%s/X_test_%s.txt" % (config.source, config.folder, iter), index_col=0, header=0, sep=",")
        y_train = pd.read_csv("output/%s/%s/y_train_%s.txt" % (config.source, config.folder, iter), index_col=0, header=0, sep=",")
        y_test = pd.read_csv("output/%s/%s/y_test_%s.txt" % (config.source, config.folder, iter), index_col=0, header=0, sep=",")

        metrics_summary = []

        for value, experiment in zip(values, models):
            print("\n", experiment)
            config.classifier = value
            config.models = experiment
            model = ModelML(config, dataset, X_train, y_train, X_test, y_test, iter)
            y_scores, y_preds = model.train_predict()
            model.get_metrics(y_scores, y_preds)
            model.plot_ROCs(y_scores)
            model.plot_PRs(y_scores)
            metrics_summary.append(np.mean(model.metrics, axis=0).values)

        metrics_summary = pd.DataFrame(data=np.array(metrics_summary), index=models, columns=model.metrics.columns)
        metrics_summary.to_csv("output/%s/%s/metrics_summary_%s.txt" % (config.source, config.folder, iter))
        config.fig.savefig("output/%s/%s/curves_%s" % (config.source, config.folder, iter))
        plt.close()
