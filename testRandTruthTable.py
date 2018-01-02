import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense

text = input("Enter a 3 digits of 0 and 1's:")
input_array = np.array([0,0,0])
i = 0
for digit in text:
	input_array[i] = digit
	i+=1;

for val in input_array:
	print(val)
