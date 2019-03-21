import seaborn as sns

def make_confusion_matrix():
	true_positive = 1337
	false_positive = 200
	true_negative = 777
	false_negative = 99
	
    matrix = [[true_positive, false_negative],[false_positive, true_negative]]
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=[],
                yticklabels=[],
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
