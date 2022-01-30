"""Tests for the keras to numpy module.""" 

import numpy 
import pytest 
import torch 
from torch import nn 

from concrete.keras import KerasNumpyModule 

def create_model(choice, input_shape, num_classes): 
	"""
	Create a base model for experiments. 
	Choice (str): specify the model 

	"""

	model = None 

	if choice == "Conv":  

		model = Sequential(
		    [
		        Input(shape=input_shape),
		        Conv2D(32, kernel_size=(3, 3), activation="relu"),
		        MaxPooling2D(pool_size=(2, 2)),
		        Conv2D(64, kernel_size=(3, 3), activation="relu"),
		        MaxPooling2D(pool_size=(2, 2)),
		        Flatten(),
		        Dropout(0.5),
		        Dense(num_classes, activation="softmax"),
		    ]
		) 

	elif choice == "Linear3Layer": 
		model = Sequential(
			[Input(shape=input_shape), 
			Flatten(),
			Dense(128, activation="relu"),
			Dense(128, activation="relu"),
			Dense(128, activation="relu"),
			Dense(num_classes, activation="softmax")



			])

	return model 


@pytest.mark.parametrize(
    "model, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)), # need to change this 
    ],
)
@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.Sigmoid, id="sigmoid"), # need to change this 
        pytest.param(nn.ReLU6, id="relu"), # need to change this 
    ],
) 

def test_keras_to_numpy(model, input_shape, activation_function, seed_keras): 
	""" Test the different model architecture from keras numpy.""" 

	# Seed torch 
	seed_keras() 
	# Define the keras model 
	keras_fc_model = None # need to change this 
	# Create random input 
	keras_input_1 = None # need to change this 
	# Predict with keras model 
	keras_predictions = keras_fc_model(keras_input_1).numpy() # need to check this 
	# Create corresponding numpy model 
	numpy_fc_model = KerasNumpyModule(keras_fc_model) 
	# Keras input to numpy 
	numpy_input_1 = keras_input_1.numpy() # need to check this 
	# Predict with numpy model 
	numpy_predictions = numpy_fc_model(numpy_input_1) 

	# Test: the output of the numpy model is the same as the torch model. 
	assert numpy_predicitons.shape == keras_predictions.shape 
	# Test: prediction from the numpy model are the same as the keras model. 
	assert numpy.isclose(keras_predictions, numpy_predictions, rtol=10 - 3).all() 

	# Test: dynamics between layers is working (quantized input and activations) 
	keras_input_2 = None # need to change this 
	# Make sure both inputs are different 
	assert (keras_input_1 != keras_input_2).any() 
	# Predict with keras 
	keras_predictions = keras_fc_model(keras_input_2).numpy() # need to check this 
	# Keras input to numpy 
	numpy_input_2 = keras_input_2.numpy() # need to check this 
	# Numpy predictions using the previous model 
	numpy_predictions = numpy_fc_model(numpy_input_2) 
	assert numpy.isclose(keras_predictions, numpy_predictions, rtol = 10 - 3).all() 


@pytest.mark.parametrize(
    "model, incompatible_layer",
    [pytest.param(CNN, "Conv2d")],
) 

def test_raises(model, incompatible_layer, seed_keras): 
	"""Function to test incompatible layers.""" 

	seed_keras() 
	keras_incompatible_model = model() # need to change this 
	expected_errmsg = ( 
		f"The following module is currently not implemented: {incompatible_layer}. " 
		f"Please stick to the available keras modules: " 
		f"{', '.join(sorted(module.__name__ for module in KerasNumpyMOdule.IMPLEMENTED_MODULES))}."
	) 
	with pytest.raises(ValueError, match=expected_errmsg): 
		KerasNumpyModule(keras_incompatible_model)  
