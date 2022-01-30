"""Tests for the keras to numpy module.""" 

import numpy as np 
import pytest 
import tensorflow as tf 
import tensorflow.keras as keras 

from concrete.quantization import QuantizedArray
from concrete.keras.compile import compile_keras_module 


# INPUT_OUTPUT_FEATURE is the number of input and output of each of the network layers.
# (as well as the input of the network itself)
INPUT_OUTPUT_FEATURE = [1, 2] 


# need to put the keras model in here 

@pytest.mark.parametrize(
	"activation", 
	[
		pytest.param(), 
		pytest.param()
	], 
) 

@pytest.mark.parametrize( 
	"model", 
	[pytest.param()], 
) 

@pytest.mark.parametrize(
	"input_output_feature", 

    [pytest.param(input_output_feature) for input_output_feature in INPUT_OUTPUT_FEATURE],
) 

def test_compile_keras( 
	input_output_feature, 
	model, 
	activation_function, 
	seed_keras, 
	default_compilation_configuration, 
	check_is_good_execution, 
): 
	"""Test the different model architecture from keras numpy.""" 

	# Seed keras 
	keras.utils.set_random_seed(50) 

	n_bits = 2 

	# Define an input shape (n_examples, n_features) 
	n_examples = 50 

	# Define the keras model 
	keras_fc_model = None # [need to build this up & slot in]
	# Create random input 
	inputset = [
        numpy.random.uniform(-100, 100, size=input_output_feature) for _ in range(n_examples)
    ] 

    # Compile 
    quantized_numpy_module = compile_keras_model( 
    	keras_fc_model, 
    	inputset, 
    	default_compilation_configuration, 
    	n_bits=n_bits, 
    ) 

    # Quantize inputs all at once to have meaningful scale and zero point 
    q_input = QuantizedArray(n_bits, np.array(inputset)) 

    # Compare predictions between FHE and QuantizedModule 
    for x_q in q_input.qvalues: 
    	x_q = np.expand_dims(x_q, 0) 
    	check_is_good_execution( 
    		fhe_circuit=quantized_numpy_module.forward_fhe, 
    		function=quantized_numpy_module.forward, 
    		args=[x_q.astype(np.uint8)], 
    		postprocess_output_func=lambda x: quantized_numpy_module.dequantize_output(
    			x.astype(np.float32)
    		), 
    		check_function=np.isclose, 
    		verbose=False, 
    		)
