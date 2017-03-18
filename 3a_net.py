from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from commons import mnist

data = mnist()

model = Sequential()
model.add(Dense(625, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.5), metrics=['accuracy'])
model.fit(data.train.images, data.train.labels, batch_size=100, epochs=100, verbose=0)
print(model.evaluate(data.test.images, data.test.labels, verbose=0)[1]) # ~98.2%