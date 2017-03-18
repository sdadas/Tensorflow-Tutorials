from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from commons import mnist

data = mnist()

model = Sequential()
model.add(Dropout(0.2, input_shape=(784,)))
model.add(Dense(625, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(625, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(data.train.images, data.train.labels, batch_size=100, epochs=100, verbose=0)
print(model.evaluate(data.test.images, data.test.labels, verbose=0)[1]) # ~98.6%