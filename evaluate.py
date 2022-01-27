import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from tkinter.filedialog import askdirectory
import pickle

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
class_names = ['ERG_Negative', 'ERG_Positive']
TP = { "10x": [0]*21, "20x": [0]*21, "40x": [0]*21}
FP = { "10x": [0]*21, "20x": [0]*21, "40x": [0]*21}
TN = { "10x": [0]*21, "20x": [0]*21, "40x": [0]*21}
FN = { "10x": [0]*21, "20x": [0]*21, "40x": [0]*21}
sensitivity = { "10x": [0]*21, "20x": [0]*21, "40x": [0]*21}
specificity = { "10x": [0]*21, "20x": [0]*21, "40x": [0]*21}


directory = askdirectory(title='Select Folder')
case_set = "wsi"
magnifications = ["10x","20x","40x"]
correct_labels = ["ERG_Negative", "ERG_Positive"]

cutoff = [i/100 for i in range(0,105,5)]
for cut in cutoff:
    index = (int)(cut*100/5)
    for magnification in magnifications:
        print(magnification)
        model = load_model(f'model')
        case_results = []
        for correct_label in correct_labels:
            PATH = os.path.join(directory, magnification, case_set, correct_label)
            for root, subdirectories, files in os.walk(PATH):
                for subdirectory in subdirectories:
                    img_dir = os.path.join(root, subdirectory)
                    img_dir = img_dir.replace(os.path.sep, '/')
                    path_split = img_dir.split("/")
                    correct_label = path_split[-2]
                    case_ID = path_split[-1]
                    BATCH_SIZE = 32
                    IMG_SIZE = 224
                    AUTOTUNE = tf.data.experimental.AUTOTUNE
                    
                    num_tiles = len([name for name in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, name).replace(os.path.sep,'/'))])

                    print(img_dir)
                    if num_tiles == 0:
                        print(img_dir)
                        continue
                    batch_holder = np.zeros((num_tiles, IMG_SIZE, IMG_SIZE, 3))

                    for i,img in enumerate(os.listdir(img_dir)):
                      img = image.load_img(os.path.join(img_dir,img), target_size=(IMG_SIZE,IMG_SIZE))
                      img = image.img_to_array(img)
                      img = np.expand_dims(img, axis=0)
                      batch_holder[i, :] = img
                      

                    pred = model.predict(batch_holder)
                    result = [int(round(p[0])) for p in pred ]

                    labels = []
                    neg = 0
                    pos = 0
                    for i,img in enumerate(batch_holder):
                      labels.append(class_names[result[i]])
                      if class_names[result[i]] == 'ERG_Negative':
                          neg += 1
                      else:
                          pos += 1

                    if(pos>=cut*(pos+neg)):
                        algorithm_label = 'ERG_Positive'
                        print(f"ERG_Positive, Tiles:{num_tiles}, Pos:{pos}, Neg:{neg}")
                    else:
                        algorithm_label = 'ERG_Negative'
                        print(f"ERG_Negative, Tiles:{num_tiles}, Pos:{pos}, Neg:{neg}")

                    agree = 1 if algorithm_label == correct_label else 0

                    if correct_label == "ERG_Positive":
                        if(algorithm_label == correct_label):
                            TP[f"{magnification}"][index] += 1
                        else:
                            FN[f"{magnification}"][index] += 1
                    else:
                        if(algorithm_label == correct_label):
                            TN[f"{magnification}"][index] += 1
                        else:
                            FP[f"{magnification}"][index] += 1    

        sensitivity[f"{magnification}"][index] = TP[f"{magnification}"][index]/(TP[f"{magnification}"][index]+FN[f"{magnification}"][index])
        specificity[f"{magnification}"][index] = TN[f"{magnification}"][index]/(FP[f"{magnification}"][index]+TN[f"{magnification}"][index])
    print(cut)

# with open('accuracy_values_10x_20x_40x.pickle', 'rb') as f:
#     TPall, FPall, TNall, FNall, accuracyall, f1all, PPVall, NPVall = pickle.load(f)
   
# with open('sensitivity_specificity_values_10x_20x_40x.pickle', 'rb') as f:
#     sensitivityall, specificityall = pickle.load(f)
    
with open('sensitivity_specificity_values_10x_20x_40x_low_grade.pickle', 'wb') as f:
    pickle.dump([sensitivity, specificity], f)

with open('sensitivity_specificity_values_10x_20x_40x_low_grade.pickle', 'rb') as f:
   sensitivity_10x_20x, specificity_10x_20x = pickle.load(f)

tpr_keras10x = sensitivity_10x_20x["10x"]
fpr_keras10x = [1-i for i in specificity_10x_20x["10x"]]
tpr_keras20x = sensitivity_10x_20x["20x"]
fpr_keras20x = [1-i for i in specificity_10x_20x["20x"]]
auc_keras10x = auc(fpr_keras10x, tpr_keras10x)
auc_keras20x = auc(fpr_keras20x, tpr_keras20x)

tpr_keras40x = sensitivity_10x_20x["40x"]
fpr_keras40x = [1-i for i in specificity["40x"]]
auc_keras40x = auc(fpr_keras40x, tpr_keras40x)

colors = iter([plt.cm.Set2(i) for i in range(9)])
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras10x, tpr_keras10x, label= '10x ' + ' (area = {:.3f})'.format(auc_keras10x), c=next(colors), linewidth=3)
plt.plot(fpr_keras20x, tpr_keras20x, label= '20x ' + ' (area = {:.3f})'.format(auc_keras20x), c=next(colors), linewidth=3)
plt.plot(fpr_keras40x, tpr_keras40x, label= '40x ' + ' (area = {:.3f})'.format(auc_keras40x), c=next(colors), linewidth=3)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='best')
plt.savefig('ROC Curves colored_low_grade.tif', dpi=300)
plt.show()

accuracy = {mag:[0]*21 for mag in magnifications}
f1 = {mag:[0]*21 for mag in magnifications}
PPV = {mag:[0]*21 for mag in magnifications}
NPV = {mag:[0]*21 for mag in magnifications}

for mag in magnifications:
    for cut in cutoff:
        index = (int)(cut*100/5)
        accuracy[mag][index] = np.divide((TP[mag][index]+TN[mag][index]),(TP[mag][index]+TN[mag][index]+FP[mag][index]+FN[mag][index]))
        f1[mag][index] = np.divide(2*TP[mag][index],(2*TP[mag][index] + FP[mag][index] + FN[mag][index]))
        PPV[mag][index] = np.divide(TP[mag][index],(TP[mag][index]+FP[mag][index]))
        NPV[mag][index] = np.divide(TN[mag][index],(TN[mag][index]+FN[mag][index]))
        

with open('accuracy_values_10x_20x_40x_low_grade.pickle', 'wb') as f:
    pickle.dump([TP, FP, TN, FN, accuracy, f1, PPV, NPV], f)

with open('accuracy_values_10x_20x_40x_low_grade.pickle', 'rb') as f:
   TP, FP, TN, FN, accuracy, f1, PPV, NPV = pickle.load(f)

results = {"Cutoffs": cutoff, "True Positives 10x": TP["10x"], "False Positives 10x": FP["10x"], "True Negatives 10x": TN["10x"], "False Negatives 10x": FN["10x"], "Sensitivity_10x": sensitivity_10x_20x["10x"], "Specificity_10x": specificity_10x_20x["10x"], "Accuracy_10x": accuracy["10x"], "F1 Score 10x": f1["10x"], "PPV 10x": PPV["10x"], "NPV 10x": NPV["10x"], 
           "True Positives 20x": TP["20x"], "False Positives 20x": FP["20x"], "True Negatives 20x": TN["20x"], "False Negatives 20x": FN["20x"],"Sensitivity_20x": sensitivity_10x_20x["20x"], "Specificity_20x":specificity_10x_20x["20x"], "Accuracy_20x": accuracy["20x"], "F1 Score 20x": f1["20x"], "PPV 20x": PPV["20x"], "NPV 20x": NPV["20x"], 
           "True Positives 40x": TP["40x"], "False Positives 40x": FP["40x"], "True Negatives 40x": TN["40x"], "False Negatives 40x": FN["40x"],"Sensitivity_40x": sensitivity["40x"], "Specificity_40x":specificity["40x"], "Accuracy_40x": accuracy["40x"], "F1 Score 40x": f1["40x"], "PPV 40x": PPV["40x"], "NPV 40x": NPV["40x"]}
df = pd.DataFrame(results)
file_name = "Sensitivity_Specificity_all_high_grade.csv"
df.to_csv(file_name, header= ["Cutoff Values", "True Positives 10x", "False Positives 10x", "True Negatives 10x", "False Negatives 10x", "sensitivity_10x", "specificity_10x", "accuracy_10x", "f1score_10x", "PPV_10x", "NPV_10x",
                              "True Positives 20x", "False Positives 20x", "True Negatives 20x", "False Negatives 20x","sensitivity_20x", "specificity_20x", "accuracy_20x", "f1score_20x", "PPV_20x", "NPV_20x",
                              "True Positives 40x", "False Positives 40x", "True Negatives 40x", "False Negatives 40x","sensitivity_40x", "specificity_40x", "accuracy_40x", "f1score_40x", "PPV_40x", "NPV_40x"] , index=False)
