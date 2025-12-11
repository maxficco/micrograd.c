#ifndef MICROGRAD_H
#define MICROGRAD_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>

typedef struct Value {
    float data;
    float grad;
    void (*grad_fn)(struct Value *self, struct Value *prev[2]);
    struct Value *prev[2]; // 1 or 2 inputs per operation (or 0 inputs for noop)
    int tape_idx; // so we know where to start backprop
    struct Value *next; // for keeping track of weights allocation on heap
} Value;

double random_uniform(double min, double max);

Value *new_val(float data, Value *prev0, Value *prev1);
Value *new_param(float data);
void print_value(Value *v);

Value *add(Value *self, Value *other);
Value *sub(Value *self, Value *other);
Value *mul(Value *self, Value *other);
Value *true_div(Value *self, Value *other);
Value *v_pow(Value *self, float n);
Value *v_div(Value *self, Value *other);
Value *v_exp(Value *self);
Value *v_tanh(Value *self);
Value *relu(Value *self);

void backward(Value *root, bool retain_graph);
void update_params(float lr);

void free_vals();
void free_params();
void zero_grad();
void zero_grad_all();

int testing();
void backward_dfs(Value *root, bool retain_graph);

#endif // MICROGRAD_H
