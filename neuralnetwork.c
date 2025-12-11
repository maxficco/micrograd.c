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

Neuron *new_neuron(int nin, Value* (*activation)(Value *self)) {
    Neuron *n = malloc(sizeof(Neuron));

    n->weights = malloc(nin*sizeof(Value*));
    for (int i = 0; i < nin; i++) {
        n->weights[i] = new_param(random_uniform(-0.5,0.5));
    }
    n->nin= nin;
    n->bias = new_param(0);

    n->activation = activation;

    return n;
}

Layer *new_layer(int nin, int nout, Value* (*activation)(Value *self)) {
    Layer *l = malloc(sizeof(Layer));

    l->neurons = malloc(nout*sizeof(Neuron*));
    l->output_buffer = malloc(nout*sizeof(Value*));
    for (int i = 0; i < nout; i++) {
        l->neurons[i] = new_neuron(nin, activation);
    }
    l->nin = nin;
    l->nout = nout;

    return l;
}

MLP *new_mlp(int inputdim, int nlayers, int *layerdims) {
    MLP *mlp = malloc(sizeof(MLP));
    mlp->layers = malloc(nlayers*sizeof(Layer*));
    mlp->nlayers = nlayers;

    int nin, nout;
    for (int i=0; i<nlayers; i++) {
        nin = (i == 0) ? inputdim : layerdims[i-1];
        nout = layerdims[i];

        Value* (*activation)(Value *self) = (i == nlayers-1) ? NULL : v_tanh;

        mlp->layers[i] = new_layer(nin, nout, activation);
    }
    return mlp;
}

//        sum  = 0
// w[0]*x[0] --> +--> sum
// w[1]*x[1] -->   +   ^--> sum
// w[2]*x[2] -->      +      ^--> sum
// w[3]*x[3] -->         +         ^--> sum
//       bias-->            +            ^--> output
//       (if activation_func!=NULL)              ^-->activation(output) --> ouput
Value* neuron_forward(Neuron *n, Value **x) {
    Value *sum = new_val(0, NULL, NULL);
    Value *wixi;
    for (int i=0; i<n->nin; i++) {
        wixi = mul(n->weights[i], x[i]);
        sum = add(sum, wixi);
    }
    
    n->output = add(sum, n->bias);

    if (n->activation != NULL) {
        n->output = n->activation(n->output);
    }

    return n->output;
}

Value** layer_forward(Layer *l, Value **x) {
    for (int i=0; i<l->nout; i++) {
        l->output_buffer[i] = neuron_forward(l->neurons[i], x);
    }
    return l->output_buffer;
}

Value** forward(MLP *mlp, Value **inputs) {
    for (int i=0; i<mlp->nlayers; i++) {
        inputs = layer_forward(mlp->layers[i], inputs);
    }
    return inputs;
}

void free_mlp(MLP *mlp) {
    for (int i=0; i<mlp->nlayers; i++) {
        Layer *l = mlp->layers[i];
        for (int j=0; j<l->nout; j++) {
            Neuron *n = l->neurons[j];
            // note that we don't free any internal value structs here:
            // - vals: live "on the tape"!
            // - params: in the heap, but tracked by parameters_head
            // ...so we keep track of these separately and deal with them below
            free(n->weights);
            free(n);
        }
        free(l->neurons);
        free(l->output_buffer);
        free(l);
    }
    free(mlp->layers);
    free(mlp);

    free_vals(); // just sets index pointer (tape_head) back to 0
    free_params(); // traverses linked list and frees all weights and biases
}

