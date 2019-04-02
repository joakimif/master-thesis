import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def heatmap():
    true_positive = 95
    false_positive = 7
    true_negative = 93
    false_negative = 5
    
    matrix = np.array([[true_positive, false_negative], [false_positive, true_negative]])
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=['Positive', 'Negative'],
                yticklabels=['Positive', 'Negative'],
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.savefig('seaborn.svg')

heatmap()