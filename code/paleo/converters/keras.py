import json

layer_map = {
    "<class 'tensorflow.python.keras.layers.convolutional.Conv2D'>": "Convolution",
    "<class 'tensorflow.python.keras.layers.pooling.MaxPooling2D'>": "Pooling",
    "<class 'tensorflow.python.keras.layers.core.Dropout'>": "Dropout",
    "<class 'tensorflow.python.keras.layers.core.Dense'>": "InnerProduct",
}


def _shape_to_list(shape):
    return [int(dim) for dim in shape]


def convert_model(model, batch_size, name='Converted Keras Model'):
    json_model = {
        'name': name,
        'layers': {}
    }

    # Add model inputs
    for input in model.inputs:
        layer = {
            'parents': [],
            'type': 'Input',
            'tensor': [batch_size] + _shape_to_list(input.shape[1:]),
        }

        json_model['layers'][input.name.split(':')[0]] = layer

    # Add all layers
    for layer in model.layers:
        try:
            typ = str(type(layer))
            layer_type = layer_map[typ]
        except KeyError:
            print(f'Could not find a mapping for layer of type {type(layer)}')
            print(f'Adding it as an input layer (which does nothing to the data)')
            layer_type = 'Input'

        layer_dict = {
            'type': layer_type,
        }

        if layer_type == 'Convolution':
            layer_dict['filter'] = _shape_to_list(layer.kernel.shape)
            layer_dict['padding'] = layer.padding.upper()
            layer_dict['strides'] = [1] + _shape_to_list(layer.strides) + [1]
        elif layer_type == 'Pooling':
            layer_dict['padding'] = layer.padding.upper()
            layer_dict['ksize'] = [1] + _shape_to_list(layer.strides) + [1]
            layer_dict['strides'] = [1] + _shape_to_list(layer.strides) + [1]
        elif layer_type == 'Dropout':
            layer_dict['dropout_keep_prob'] = 1 - layer.rate
        elif layer_type == 'InnerProduct':
            layer_dict['num_outputs'] = layer.units
        elif layer_type == 'Input':
            layer_dict['tensor'] = [batch_size] + _shape_to_list(layer.output.shape[1:])

        layer_dict['parents'] = [layer.input.name.replace(':', '/').split('/')[0]]

        json_model['layers'][layer.name] = layer_dict

    return json.dumps(json_model, indent=4)