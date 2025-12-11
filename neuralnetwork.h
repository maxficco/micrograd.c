#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "micrograd.h"

typedef struct Neuron {
    int nin;
    Value **weights;
    Value *bias;
    Value *output; // just for potential debug
    Value* (*activation)(Value *self);
} Neuron;

typedef struct Layer {
    int nin;
    int nout;
    Neuron **neurons;
    Value **output_buffer;
} Layer;

typedef struct MLP {
    int nlayers;
    Layer **layers;
} MLP;

Neuron *new_neuron(int nin, Value* (*activation)(Value *self));
Layer *new_layer(int nin, int nout, Value* (*activation)(Value *self));
MLP *new_mlp(int inputdim, int nlayers, int *layerdims);

Value* neuron_forward(Neuron *n, Value **x);
Value** layer_forward(Layer *l, Value **x);
Value** forward(MLP *mlp, Value **inputs);

void free_mlp(MLP *mlp);

#endif // NEURALNETWORK_H
