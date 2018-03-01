from tree import DecisionTreeClassifier
import pandas as pd # optional

clf = DecisionTreeClassifier()

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

data = pd.DataFrame(X, Y, columns=['Height', 'Weight', 'Foot Size'])
print(data)

clf = clf.fit(X, Y)

questions = [[190, 70, 43], [175, 55, 40]]
predictions = clf.predict(questions)

print('\nPredictions:')
for i, prediction in enumerate(predictions):
    print(questions[i], prediction)