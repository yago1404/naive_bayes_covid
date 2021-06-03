import pickle
from training_model import training

print('Responda as perguntas com 1 ou 0 (sim ou não)\n')

fever = int(input('Você esta com febre?\n=>'))
tiredness = int(input('Você sente muito cansaço?\n=>'))
dry_cough = int(input('Você esta com tosse seca?\n=>'))
dificulty_in_breath = int(input('Esta sentindo dificuldade para respirar?\n=>'))
sore_throat = int(input('Sente dor de garganta?\n=>'))
none_sympton = int(input('Esta sem sintomas?\n=>'))
paints = int(input('Você esta sentindo dores?\n=>'))
nasal_congestion = int(input('Seu nariz está entupido?\n=>'))

data = [fever, tiredness, dry_cough, dificulty_in_breath, sore_throat, none_sympton, paints, nasal_congestion]
try:
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
except:
    training()
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()


print('\nSegundo a análise do modelo, a propabilidade de ter complicações da doença são => ', classifier.predict([data])[0])