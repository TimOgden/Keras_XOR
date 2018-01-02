import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense

#the four different states of the XOR gate
training_data = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
			[1,0,0],[1,0,1],[1,1,0],[1,1,1]], "float32")

#the four expected results in the same order
target_data = np.array([[0],[1],[0],[1],[0],[1],[1],[1]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
			optimizer='adam',
			metrics=['binary_accuracy'])
model.fit(training_data, target_data, nb_epoch=500, verbose=2)

print(model.predict(training_data).round())

#save model
model_yaml = model.to_yaml()
with open('model.yaml', 'w') as yaml_file:
	yaml_file.write(model_yaml)
	
#save the weights
model.save_weights('model_weights.h5')
print('Saved model and weights to disk')