# Keras Test

This is a convolutional neural network built using Keras. It uses a relu activation function on the input layer and a softmax activation function on the output layer. The loss is computed by using the cross entropy function. It is used on the Yalefaces dataset and categorizes each face into 1 of 14 different classes.

## Usage

To generate the pickle data:

`python3 gen_data.py`

To run the training program:

`python3 keras_test.py`

## Dependencies

- `python 3.8+`

### Python Dependencies

- `pickle`
- `numpy`
- `keras`

## Hyper Parameters

- Learning Rate: `0.001`

- Termination Criteria: `15 iterations`

- Batch Size: `8`

## Results

### Network

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 36, 36, 128)       3328
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 18, 18, 128)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        204864
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 3136)              0
_________________________________________________________________
dense (Dense)                (None, 32)                100384
_________________________________________________________________
dense_1 (Dense)              (None, 16)                528
=================================================================
```

### Accuracy

Validation Accuracy: `91.89%`

Testing Accuracy: `90.48%`

### Loss

![](https://github.com/is386/KerasTest/blob/master/loss.png?raw=true)
