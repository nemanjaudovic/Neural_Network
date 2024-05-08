import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

data = pd.read_csv("data.csv", header=None)
data.columns = ['x1','x2', 'y']

ulaz = data.iloc[:,:2].to_numpy()
izlaz = data.y.to_numpy()


klase = np.unique((izlaz))



K1 = ulaz[izlaz == klase[0],:]
K2 = ulaz[izlaz == klase[1],:]
K3 = ulaz[izlaz == klase[2],:]



"""plt.figure()
plt.plot(K1[:,0], K1[:,1], 'o')
plt.plot(K2[:,0], K2[:,1],'*')
plt.plot(K3[:,0], K3[:,1], 'd')
plt.show()"""

izlazOH = to_categorical(izlaz)

ulazTrening, ulazTest, izlazTreningOH, izlazTestOH = train_test_split(ulaz, izlazOH, test_size=0.2,random_state=42)

ulazTrening, ulazVal, izlazTreningOH, izlazValTreningOH = train_test_split(ulazTrening, izlazTreningOH,
                                                                           test_size=0.8, random_state=42)


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(200, input_dim=np.shape(ulazTrening)[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(np.shape(izlazTreningOH)[1], activation='softmax'))

print(model.summary())

# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
#
# es = EarlyStopping(monitor = 'val_loss', mode='min', patience = 20, verbose = 1, restore_best_weights=True)
#
#
# history = model.fit(ulazTrening, izlazTreningOH, epochs = 1000, batch_size=64, validation_data=(ulazVal, izlazValTreningOH), verbose = 0)
#
# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.show()
#
# izlazTreningPredOH = model.predict(ulazTrening, verbose=0)
# izlazTreningPred = np.argmax(izlazTreningPredOH, axis=1)
# izlazTrening = np.argmax(izlazTreningOH, axis=1)
#
# Atrening = np.sum(izlazTreningPred == izlazTrening)/len(izlazTrening)
# print('Tacnost na trening skupu iznosi: ' + str(Atrening*100) + "%.")
#
# izlazTestPredOH = model.predict(ulazTest, verbose=0)
# izlazTestPred = np.argmax(izlazTestPredOH, axis = 1)
# izlazTest = np.argmax(izlazTestOH, axis = 1)
# Atest = np.sum(izlazTestPred == izlazTest)/len(izlazTest)
# print("tacnost na test skupu je " + str(Atest*100) + " %.")





