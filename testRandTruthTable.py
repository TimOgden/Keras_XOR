import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.models import model_from_yaml

text = input("Enter a 3 digits of 0 and 1's:")
input_array = np.array([0,0,0])
i = 0
for digit in text:
	input_array[i] = digit
	i+=1;

#Define number of classes
num_classes = 2
#Input dimensions
rows, cols = 3, 1


#load model yaml
with open('model.yaml', 'r') as yaml_file:
	loaded_yaml_model = yaml_file.read()

#Create model from yaml config
loaded_model = model_from_yaml(loaded_yaml_model)

#Load weights from trained network
loaded_model.load_weights('model_weights.h5')
print("Loaded model and weights from disk")

prediction = loaded_model.predict(np.array([[input_array[0],input_array[1],
	input_array[2]]]), batch_size=1, verbose=2)
print('Prediction: {} with {:.2f}% confidence.'.format(np.argmax(prediction), np.max(prediction)*100))