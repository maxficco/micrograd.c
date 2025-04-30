#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

typedef struct Value {
   float data;
   float grad;
   void (*backward)(struct Value *self, struct Value *children[2]);
   struct Value *prev[2]; // 1 or 2 children per operation (or 0 children for noop)
   int visited; // for topo sort
   char label;
   int mallocated;  // was this node heap-allocated?
} Value;

void print_value(Value *v) {
    printf("Value(data=%lf, grad=%lf) - '%c'\n", v->data, v->grad, v->label);
}

void noop_backward(Value *self, Value *children[2]) {
    // do nothing (leaf nodes), default
}

void add_backward(Value *self, Value *children[2]) {
    children[0]->grad += 1.0 * self->grad; // think chain rule, local derivative is 1 for addition
    children[1]->grad += 1.0 * self->grad;
}
void mul_backward(Value *self, Value *children[2]) {
    children[0]->grad += children[1]->data * self->grad; // think chain rule, dz/dx = (dz/du)*(du/dx), local derivative du/dx is the coefficient
    children[1]->grad += children[0]->data * self->grad;
}

void div_backward(Value *self, Value *children[2]) {
    float x = children[0]->data;
    float y = children[1]->data;
    // z = x / y, dz/dx = 1/y, dz/dy = -x / y^2
    children[0]->grad += (1.0 / y) * self->grad;
    children[1]->grad += (-x / (y*y)) * self->grad;
}

void pow_backward(Value *self, Value *children[2]) {
    assert(children[0] != NULL); // we assume only child occupies index 0
    float n = children[1]->data; // scalar exponent passed as Value object, but not treated as such
    children[0]->grad += (n * pow(children[0]->data, n-1)) * self->grad; // local derivative of x^n is n*x^n-1
}

void exp_backward(Value *self, Value *children[2]) {
    assert(children[0] != NULL && children[1] == NULL); // we assume only child occupies index 0 
    children[0]->grad += self->data * self->grad; // derivative of e^x is e^x
}

void tanh_backward(Value *self, Value *children[2]) {
    assert(children[0] != NULL && children[1] == NULL); // we assume only child occupies index 0
    children[0]->grad += (1 - (self->data * self->data)) * self->grad; // local derivative of tanh * gradient (chain rule again)
}

void relu_backward(Value *self, Value *children[2]) {
    assert(children[0] != NULL && children[1] == NULL); // we assume only child occupies index 0
    float x = children[0]->data;
    // relu is y=x for x>0, and y=0 for x <= 0
    // => local derivative is 1 for x>0, and 0 for x <= 0
    children[0]->grad += (x > 0) * self->grad;
}


Value newValue(float data, Value *children[2]) {
    Value v;
    v.data = data;
    v.grad = 0.0;
    v.backward = noop_backward;
    if (children != NULL) {
        if (children[0] != NULL) {
            v.prev[0] = children[0];
        } else {
            v.prev[0] = NULL;
        }
        if (children[1] != NULL) {
            v.prev[1] = children[1];
        } else {
            v.prev[1] = NULL;
        }
    } else { // leaf node
        v.prev[0] = NULL;
        v.prev[1] = NULL;
    }
    v.visited = 0;
    v.label = ' ';
    v.mallocated = 0;
    return v;
}

Value add(Value *self, Value *other) {
    Value *children[2] = {self, other};
    Value out = newValue(self->data + other->data, children);

    out.backward = add_backward;

    return out;
}

Value mul(Value *self, Value *other) {
    Value *children[2] = {self, other};
    Value out = newValue(self->data * other->data, children);

    out.backward = mul_backward;

    return out;
}

Value true_div(Value *self, Value *other) {
    Value *children[2] = {self, other};
    Value out = newValue(self->data / other->data, children);

    out.backward = div_backward;

    return out;
}

Value v_pow(Value *self, float n) { // Value to a scalar power
    Value *exponent = malloc(sizeof(Value)); // avoids dangling pointer!
    *exponent = newValue(n, NULL);
    exponent->mallocated = 1; // keep track - need to free later!

    Value *children[2] = {self, exponent};
    Value out = newValue(pow(self->data, n), children);

    out.backward = pow_backward;

    return out;
}

// idea: a/b = a*(b**-1)
Value v_div(Value *self, Value *other) {
    Value *recip = malloc(sizeof(Value)); // avoids dangling pointer!
    *recip = v_pow(other, -1.0f);
    recip->mallocated = 1; // keep track - need to free later!

    return mul(self, recip);
}

Value v_exp(Value *self) {
    Value *children[2] = {self, NULL}; // only child
    float x = self->data;
    Value out = newValue(exp(x), children);

    out.backward = exp_backward;

    return out;
}

Value v_tanh(Value *self) {
    Value *children[2] = {self, NULL}; // only child
    float x = self->data;
    Value out = newValue((exp(2*x)-1)/(exp(2*x)+1), children);

    out.backward = tanh_backward;

    return out;
}

Value relu(Value *self) {
    Value *children[2] = {self, NULL}; // only child
    float x = self->data;
    float y = (x > 0) ? x : 0;
    Value out = newValue(y, children);

    out.backward = relu_backward;

    return out;
}

// recursively counts number of total children
int count_graph(Value *v) { 
    if (v == NULL || v->visited) return 0; // base case

    v->visited = 1;  // use visited to avoid double-counting

    return 1 + count_graph(v->prev[0]) + count_graph(v->prev[1]); // add up total children of each child
}

// reset all child values to unvisited
void clear_visited(Value *v) {
    if (v == NULL || !v->visited) return; // base case

    v->visited = 0;

    clear_visited(v->prev[0]);
    clear_visited(v->prev[1]);
}

int build_topo(Value **topo, int i, Value *v) { // returns number of values added to topo so far
    if (v->visited) return i;
    v->visited = 1; // mark parent node
    if (v->prev[0] != NULL) i = build_topo(topo, i, v->prev[0]);
    if (v->prev[1] != NULL) i = build_topo(topo, i, v->prev[1]);
    topo[i] = v;
    return i+1;
}

void free_graph(Value *v) {
    if (v == NULL || !v->visited) return;

    v->visited = 1;

    free_graph(v->prev[0]);
    free_graph(v->prev[1]);

    if (v->mallocated) free(v);
}

void zero_grad(Value *v) {
// TODO
}


void backprop(Value *output_node) {
    // first count how many total children (and set all visited to 0/false)
    int count = count_graph(output_node);
    printf("Total nodes in graph: %d\n", count);
    clear_visited(output_node);

    // array of pointers to Values w/ length of total connected Values
    Value **topo = malloc(sizeof(Value *) * count);

    // build topo sort recursively
    int built = build_topo(topo, 0, output_node);
    assert(built == count); // sanity check
    clear_visited(output_node);

    // backpropogate in reverse topological order
    output_node->grad = 1.0; // don't forget!
    for (int i = count-1; i >= 0; i--) {
        topo[i]->backward(topo[i], topo[i]->prev);
        print_value(topo[i]);
    }
    free(topo);
}

int main() {
    // testing!
    Value f = newValue(6.71, NULL);
    f.label = 'f';
    Value a = relu(&f);
    a.label = 'a';
    Value b = v_tanh(&a);
    b.label = 'b';
    Value c = newValue(3.0, NULL);
    c.label = 'c';
    Value d = add(&c, &b);
    d.label = 'd';
    Value e = v_div(&f, &c);
    e.label = 'e';

    Value x = mul(&d, &e);
    x.label = 'x';


    Value y = v_pow(&x, 2);
    y.label = 'y';
    backprop(&y);


    free_graph(&y);
    return 0;
}
