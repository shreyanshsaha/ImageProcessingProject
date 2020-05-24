
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.transform import resize
import cv2
import os

def load_image_files(container_path, dimension=(128, 128)):
  """
  Load image files with categories as subfolder names 
  which performs like scikit-learn sample dataset
  
  Parameters
  ----------
  container_path : string or unicode
      Path to the main folder holding one subfolder per category
  dimension : tuple
      size to which image are adjusted to
      
  Returns
  -------
  Bunch
  """
  image_dir = Path(container_path)
  folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
  categories = [fo.name for fo in folders]

  descr = "A image classification dataset"
  images = []
  flat_data = []
  target = []
  for i, direc in enumerate(folders):
    print("Folder: ", direc)
    n = len(os.listdir(direc))
    it = 0
    for file in direc.iterdir():
      if it%100==0:
        print("{} of {} done...".format(it, n))
      it+=1
      img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

      # img_resized = img
      img_resized = resize(img, dimension, anti_aliasing=False, mode='reflect')
      flat_data.append(img_resized.flatten()) 
      images.append(img_resized)
      target.append(i)
    print("{} of {} done...".format(n, n))
  flat_data = np.array(flat_data)
  target = np.array(target)
  images = np.array(images)

  return Bunch(data=flat_data,
                target=target,
                target_names=categories,
                images=images,
                DESCR=descr)

image_dataset = load_image_files("./chest-xray-pneumonia/chest_xray/train")
X_train, _, y_train, _ = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.0001, random_state=42)

image_test_dataset = load_image_files("./chest-xray-pneumonia/chest_xray/test")
X_test, _, y_test, _ = train_test_split(
    image_test_dataset.data, image_test_dataset.target, test_size=0.0001, random_state=42)

print(X_train.shape, y_train.shape)

# param_grid = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
clf = svm.SVC(C=1.0, break_ties=False, cache_size=200,
                           class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='scale', kernel='linear', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False)
# clf = GridSearchCV(svc, param_grid)

print("Training...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))



"""
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False):
              precision    recall  f1-score   support

           0       0.96      0.35      0.51       234
           1       0.72      0.99      0.83       389

    accuracy                           0.75       623
   macro avg       0.84      0.67      0.67       623
weighted avg       0.81      0.75      0.71       623
"""