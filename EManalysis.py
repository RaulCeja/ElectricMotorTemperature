# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:20:57 2019

@author: Dr.C
"""
#https://www.kaggle.com/wkirgsn/electric-motor-temperature
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
#for predictive modelling
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
data = pd.read_csv("pmsm_temperature_data.csv")
print (data.columns)
data.columns = ["Ambient Temperature", "Coolant Temperature", 
                "Voltage-d", "Voltage-u", "Motor Speed",
                "Torque", "Current-d", "Current-q", 
                "Permanent Magnet Surface Temperature", 
                "Stator Yoke Temperature", 
                "Stator Tooth Temperature", "Stator Winding Temperature",
                "Profile ID"]
#profile ID is the session the measurement was taken. They're independent

#counts the number of unique profiles there are(without counting them all mandually) 
#and the number of rows they each have
print (data["Profile ID"].value_counts())
#segregate by profile ID since each profile is independent
p4 = data[data["Profile ID"] ==4]
print (p4.head())

p20 = data[data["Profile ID"] == 20]
p6 = data[data["Profile ID"] == 6]
p65 = data[data["Profile ID"] == 65]
p66 = data[data["Profile ID"] == 66]
p27 = data[data["Profile ID"] == 27]
p58 = data[data["Profile ID"] == 58]
p56 = data[data["Profile ID"] == 56]
p53 = data[data["Profile ID"] == 53]
p79 = data[data["Profile ID"] == 79]
p44 = data[data["Profile ID"] == 44]
p70 = data[data["Profile ID"] == 70]
p62 = data[data["Profile ID"] == 62]
p30 = data[data["Profile ID"] == 30]
p80 = data[data["Profile ID"] == 80]
p74 = data[data["Profile ID"] == 74]
p68 = data[data["Profile ID"] == 68]
p36 = data[data["Profile ID"] == 36]
p76 = data[data["Profile ID"] == 76]
p48 = data[data["Profile ID"] == 48]
p29 = data[data["Profile ID"] == 29]
p32 = data[data["Profile ID"] == 32]
p81 = data[data["Profile ID"] == 81]
p45 = data[data["Profile ID"] == 45]
p42 = data[data["Profile ID"] == 42]
p73 = data[data["Profile ID"] == 73]
p41 = data[data["Profile ID"] == 41]
p63 = data[data["Profile ID"] == 63]
p31 = data[data["Profile ID"] == 31]
p69 = data[data["Profile ID"] == 69]
p72 = data[data["Profile ID"] == 72]
p10 = data[data["Profile ID"] == 10]
p71 = data[data["Profile ID"] == 71]
p77 = data[data["Profile ID"] == 77]
p60 = data[data["Profile ID"] == 60]
p61 = data[data["Profile ID"] == 61]
p57 = data[data["Profile ID"] == 57]
p75 = data[data["Profile ID"] == 75]
p67 = data[data["Profile ID"] == 67]
p49 = data[data["Profile ID"] == 49]
p50 = data[data["Profile ID"] == 50]
p55 = data[data["Profile ID"] == 55]
p54 = data[data["Profile ID"] == 54]
p78 = data[data["Profile ID"] == 78]
p43 = data[data["Profile ID"] == 43]
p11 = data[data["Profile ID"] == 11]
p59 = data[data["Profile ID"] == 59]
p51 = data[data["Profile ID"] == 51]
p64 = data[data["Profile ID"] == 64]
p52 = data[data["Profile ID"] == 52]
p46 = data[data["Profile ID"] == 46]
p47 = data[data["Profile ID"] == 47]
#took too long, find a way to loop through value_counts

#heatmap for profile 4
sb.set(font_scale = 1.3)  #increases font size so I don't have to squint
plt.figure(figsize=(16,12))
sb.heatmap(p4.corr(),annot=True,cmap='YlGnBu',fmt='.2f',linewidths=2)
plt.show()
plt.ioff()

#heatmap for profile 54
sb.set(font_scale = 1.3)  #increases font size so I don't have to squint
plt.figure(figsize=(16,12))
sb.heatmap(p54.corr(),annot=True,cmap='YlGnBu',fmt='.2f',linewidths=2)
plt.show()
plt.ioff()

#heatmap for profile 76
sb.set(font_scale = 1.3)  #increases font size so I don't have to squint
plt.figure(figsize=(16,12))
sb.heatmap(p76.corr(),annot=True,cmap='YlGnBu',fmt='.2f',linewidths=2)
plt.show()
plt.ioff()

#plot torque and rotor temps for profile 4
#adjust size
plt.figure(figsize=(18,14), linewidth ='0.125')
plt.plot(p4['Torque'], color ='blue', label = 'Torque' )
plt.plot(p4['Motor Speed'], color ='black', label = 'Motor Speed' )
plt.plot(p4['Permanent Magnet Surface Temperature'], color = 'red', label='Permanent Magnet Surface Temperature')
plt.plot(p4['Stator Yoke Temperature'], color = 'green',label='Stator Yoke Temperature')
plt.plot(p4['Stator Tooth Temperature'], color = 'orange',label='Stator Tooth Temperature')
plt.plot(p4['Stator Winding Temperature'], color = 'purple',label='Stator Winding Temperature')
#add legend
plt.legend(loc='upper right')
#add title
plt.title('Torque, RPM, and Rotor Temperatures', fontsize=16, fontweight='bold')
plt.suptitle('Profile 4', fontsize = 14)
plt.xlabel('Time')
plt.show()
plt.ioff()

#plot torque and rotor temps for profile 6
#adjust size
plt.figure(figsize=(18,14), linewidth ='0.125')
plt.plot(p6['Torque'], color ='blue', label = 'Torque' )
plt.plot(p6['Motor Speed'], color ='black', label = 'Motor Speed' )
plt.plot(p6['Permanent Magnet Surface Temperature'], color = 'red', label='Permanent Magnet Surface Temperature')
plt.plot(p6['Stator Yoke Temperature'], color = 'green',label='Stator Yoke Temperature')
plt.plot(p6['Stator Tooth Temperature'], color = 'orange',label='Stator Tooth Temperature')
plt.plot(p6['Stator Winding Temperature'], color = 'purple',label='Stator Winding Temperature')
#add legend
plt.legend(loc='upper right')
#add title
plt.title('Torque, RPM, and Rotor Temperatures', fontsize=16, fontweight='bold')
plt.suptitle('Profile 6', fontsize = 14)
plt.xlabel('Time')
plt.show()
plt.ioff()

#plot torque and rotor temps for profile 10
#adjust size
plt.figure(figsize=(18,14), linewidth ='0.125')
plt.plot(p10['Torque'], color ='blue', label = 'Torque' )
plt.plot(p10['Motor Speed'], color ='black', label = 'Motor Speed' )
plt.plot(p10['Permanent Magnet Surface Temperature'], color = 'red', label='Permanent Magnet Surface Temperature')
plt.plot(p10['Stator Yoke Temperature'], color = 'green',label='Stator Yoke Temperature')
plt.plot(p10['Stator Tooth Temperature'], color = 'orange',label='Stator Tooth Temperature')
plt.plot(p10['Stator Winding Temperature'], color = 'purple',label='Stator Winding Temperature')
#add legend
plt.legend(loc='upper right')
#add title
plt.title('Torque, RPM, and Rotor Temperatures', fontsize=16, fontweight='bold')
plt.suptitle('Profile 10', fontsize = 14)
plt.xlabel('Time')
plt.show()
plt.ioff()
'''
#plot torque vs temps for all 
#adjust size
plt.figure(figsize=(200,16), linewidth ='0.125')
plt.plot(data['Torque'], color ='blue', label = 'Torque' )
plt.plot(data['Motor Speed'], color ='black', label = 'Motor Speed' )
plt.plot(data['Permanent Magnet Surface Temperature'], color = 'red', label='Permanent Magnet Surface Temperature')
plt.plot(data['Stator Yoke Temperature'], color = 'green',label='Stator Yoke Temperature')
plt.plot(data['Stator Tooth Temperature'], color = 'orange',label='Stator Tooth Temperature')
plt.plot(data['Stator Winding Temperature'], color = 'purple',label='Stator Winding Temperature')
#add legend
plt.legend(loc='upper right')
#add title
plt.title('Torque, RPM, and Rotor Temperatures', fontsize=16, fontweight='bold')
#plt.suptitle('Profile 10', fontsize = 14)
plt.xlabel('Time')
plt.show()
plt.ioff()
'''
# plot coolant temp and PMST in profile 6
plt.figure(figsize=(18,14))
plt.plot(p6['Coolant Temperature'],color = 'blue', label = 'Coolant Temperature')
plt.plot(p6['Permanent Magnet Surface Temperature'], color = 'red', label= 'Permanent Magnet Surface Temperature')
plt.plot(p6['Torque'], color ='green', label = 'Torque')
#add legend
plt.legend(loc = 'upper right')
#add title and axes labels
plt.title('Coolant Temperature vs Magnet Surface Temperature', fontsize = 16, fontweight = 'bold')
plt.suptitle('Profile 6', fontsize = 14)
plt.xlabel('Time')
plt.show()
plt.ioff()

#same as above but with profile 10
plt.figure(figsize=(18,14))
plt.plot(p10['Coolant Temperature'],color = 'blue', label = 'Coolant Temperature')
plt.plot(p10['Permanent Magnet Surface Temperature'], color = 'red', label= 'Permanent Magnet Surface Temperature')
plt.plot(p10['Torque'], color ='green', label = 'Torque')
#add legend
plt.legend(loc = 'upper right')
#add title and axes labels
plt.title('Coolant Temperature vs Magnet Surface Temperature', fontsize = 16, fontweight = 'bold')
plt.suptitle('Profile 10', fontsize = 14)
plt.xlabel('Time')
plt.show()
plt.ioff()

#same as above but with profile 4
plt.figure(figsize=(18,14))
plt.plot(p4['Coolant Temperature'],color = 'blue', label = 'Coolant Temperature')
plt.plot(p4['Permanent Magnet Surface Temperature'], color = 'red', label= 'Permanent Magnet Surface Temperature')
plt.plot(p4['Torque'], color ='green', label = 'Torque')
#add legend
plt.legend(loc = 'upper right')
#add title and axes labels
plt.title('Coolant Temperature vs Magnet Surface Temperature', fontsize = 16, fontweight = 'bold')
plt.suptitle('Profile 4', fontsize = 14)
plt.xlabel('Time')
plt.show()
plt.ioff()

#compare voltage, current, and performance
"Voltage-d", "Voltage-u", "Motor Speed",
"Torque", "Current-d", "Current-q"
plt.figure(figsize=(18,14))
plt.plot(p4['Torque'],color ='blue', label = 'Torque')
plt.plot(p4['Motor Speed'], color ='red', label='RPM')
plt.plot(p4['Voltage-d'], color = 'green', label = 'Voltage-d')
plt.plot(p4['Current-d'], color ='cyan', label='Current-d')
plt.title('Performance and Power Draw', fontsize=16,fontweight='bold')
plt.xlabel('Time')
plt.suptitle('Profile 4', fontsize = 14)
plt.legend(loc='upper right')
plt.show()
plt.ioff()

# histograms
data.hist(figsize = (20,20))
plt.show()

#get a better look at profile 4
p4.hist(figsize = (26,26))
plt.show()
plt.ioff()

#build machine learning models and algorithms
# Split-out validation dataset for profile 4
array = p4.values
X = array[:,0:11]  #use 11 because profile ID is unnecessary
X = np.delete(X, 8, axis=1)
Y = array[:,8] #want PMT
print(X)
#print(Y)
"""
validation_size = 0.20
seed = 5
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=seed)
#test options and evaluation metric
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
"""