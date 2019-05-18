import matplotlib.pyplot as plt
import numpy as np


def plot_hist(path, title, hist1, hist2, lbl1, lbl2, dict, save=True, suptitle=None):
    plt.hist([hist1, hist2], bins=np.arange(14) - 0.5, label=[lbl1, lbl2])
    plt.xlim([-1, 13])
    plt.legend()
    keys = list(dict.keys())
    values = list(dict.values())
    plt.xticks(keys, values, rotation='vertical')
    plt.ylabel('number of votes')
    plt.xlabel('party name')
    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.title(title)
    fig = plt.gcf()
    fig.savefig(path, bbox_inches='tight')
    plt.show()
