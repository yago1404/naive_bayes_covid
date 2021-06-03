import pickle
from training_model import training


try:
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
except:
    training()
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()


print(classifier.predict([[1,1,0,0,0,0,1,1]]))