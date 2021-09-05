import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

true_tables = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_tables = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Create confusion matrix
confusion_mat = confusion_matrix(true_tables, pred_tables)

# Visualize confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion Matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True Labels')
plt.xlabel('Predicted labels')
plt.show()

# Classification report
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4', ]
print('\n', classification_report(true_tables, pred_tables, target_names=targets))
