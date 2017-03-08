from sklearn import tree
#1 for smooth and 0 for bumpy
features=[[140,1],[130,1],[150,0],[170,0],[145,1],[138,0]]
#0 for apple and 1 for orange
labels=[0,0,1,1,1,0]
clf=tree.DecisionTreeClassifier()
clf.fit(features,labels)
print clf.predict([[149,1]])