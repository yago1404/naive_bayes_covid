import pickle
from training_model import training
import sys
import ast


data = ast.literal_eval(sys.argv[1])

try:
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
except:
    training()
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()


print(classifier.predict([data]))