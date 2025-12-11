#include "micrograd.h"

#define MAX_TAPE_SIZE 100000

static Value *parameters_head= NULL;
static Value tape_memory[MAX_TAPE_SIZE];
static int tape_head = 0;

double random_uniform(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

static void noop_backward(Value *self, Value *prev[2]) {
    // do nothing (leaf nodes), default
}

static void add_backward(Value *self, Value *prev[2]) {
    prev[0]->grad += 1.0 * self->grad; // think chain rule, local derivative is 1 for addition
    prev[1]->grad += 1.0 * self->grad;
}

static void sub_backward(Value *self, Value *prev[2]) {
    prev[0]->grad += 1.0 * self->grad;
    prev[1]->grad -= 1.0 * self->grad;
}

static void mul_backward(Value *self, Value *prev[2]) {
    prev[0]->grad += prev[1]->data * self->grad; // think chain rule, dz/dx = (dz/du)*(du/dx), local derivative du/dx is the coefficient
    prev[1]->grad += prev[0]->data * self->grad;
}

static void div_backward(Value *self, Value *prev[2]) {
    float x = prev[0]->data;
    float y = prev[1]->data;
    // z = x / y, dz/dx = 1/y, dz/dy = -x / y^2
    prev[0]->grad += (1.0 / y) * self->grad;
    prev[1]->grad += (-x / (y*y)) * self->grad;
}

static void pow_backward(Value *self, Value *prev[2]) {
    float x = prev[0]->data;
    float n = prev[1]->data;
    prev[0]->grad += (n * pow(x, n-1)) * self->grad; // local derivative of x^n is n*x^n-1 (uses math pow)
}

static void exp_backward(Value *self, Value *prev[2]) {
    assert(prev[0] != NULL && prev[1] == NULL); // we assume only child occupies index 0 
    prev[0]->grad += self->data * self->grad; // derivative of e^x is e^x
}

static void tanh_backward(Value *self, Value *prev[2]) {
    assert(prev[0] != NULL && prev[1] == NULL); // we assume only child occupies index 0
    prev[0]->grad += (1 - (self->data * self->data)) * self->grad; // local derivative of tanh * gradient (chain rule again)
}

static void relu_backward(Value *self, Value *prev[2]) {
    assert(prev[0] != NULL && prev[1] == NULL); // we assume only child occupies index 0
    float x = prev[0]->data;
    // relu is y=x for x>0, and y=0 for x<=0
    // => local derivative is 1 for x>0, and 0 for x<=0
    prev[0]->grad += (x > 0) * self->grad;
}

Value *new_param(float data) {
    Value *v = malloc(sizeof(Value));
    v->next = parameters_head;
    parameters_head= v;
    v->tape_idx = -1;

    v->data = data;
    v->grad = 0.0;
    v->grad_fn= noop_backward;
    v->prev[0] = NULL;
    v->prev[1] = NULL;
    return v;
}

Value *new_val(float data, Value *prev0, Value *prev1) {
    if (tape_head > MAX_TAPE_SIZE) {
        fprintf(stderr, "Error: Tape size exceeded!\n");
        exit(1);
    }
    Value *v = &tape_memory[tape_head];
    v->tape_idx = tape_head;
    tape_head++;
    v->data = data;
    v->grad = 0.0;
    v->grad_fn= noop_backward;
    v->prev[0] = prev0;
    v->prev[1] = prev1;
    return v;
}

Value *add(Value *self, Value *other) {
    Value *out = new_val(self->data + other->data, self, other);
    out->grad_fn = add_backward;
    return out;
}

Value *sub(Value *self, Value *other) {
    Value *out = new_val(self->data - other->data, self, other);
    out->grad_fn = sub_backward;
    return out;
}

Value *mul(Value *self, Value *other) {
    Value *out = new_val(self->data * other->data, self, other);
    out->grad_fn = mul_backward;
    return out;
}

Value *true_div(Value *self, Value *other) {
    Value *out = new_val(self->data / other->data, self, other);
    out->grad_fn = div_backward;
    return out;
}

Value *v_pow(Value *self, float n) { // Value to a scalar power
    Value *exponent = new_val(n, NULL, NULL);
    Value *out = new_val(pow(self->data, n), self, exponent);
    out->grad_fn = pow_backward;
    return out;
}

// idea: a/b = a*(b**-1)
Value *v_div(Value *self, Value *other) {
    Value *reciprocal = v_pow(other, -1.0f);
    return mul(self, reciprocal);
}

Value *v_exp(Value *self) {
    float x = self->data;
    Value *out = new_val(exp(x), self, NULL);
    out->grad_fn = exp_backward;
    return out;
}

Value *v_tanh(Value *self) {
    float x = self->data;
    Value *out = new_val((exp(2*x)-1)/(exp(2*x)+1), self, NULL);
    out->grad_fn = tanh_backward;
    return out;
}

Value *relu(Value *self) {
    float x = self->data;
    float y = (x > 0) ? x : 0;
    Value *out = new_val(y, self, NULL);
    out->grad_fn = relu_backward;
    return out;
}

// "free" all values allocated "on the tape" (our big block of Value structs allocated in data segment)
void free_vals() { 
    tape_head = 0;
    // that's it!
}

// free all values allocated on the heap
void free_params() {
    Value *next; 
    while (parameters_head != NULL) {
        next = parameters_head->next;
        free(parameters_head);
        parameters_head = next;
    }
}

// sets the gradients of all parameters to 0
void zero_grad() {
    Value *curr = parameters_head; 
    while (curr != NULL) {
        curr->grad = 0;
        curr= curr->next;
    }
}

// in case we want to retain graph (need to zero non-parameter values instead of just freeing them)
void zero_grad_all() {
    zero_grad();
    for (int i=0; i<=tape_head; i++) {
        tape_memory[i].grad = 0;
    }
}

void backward(Value *root, bool retain_graph) {
    root->grad = 1.0; // don't forget!

    // backpropagate in reverse topological order
    // WE DON'T NEED TO DO TOPOLOGICAL SORT!
    // THE TAPE DOES THIS FOR US, BECAUSE VALUES ARE STORED BY ORDER OF CREATION!
    for (int i=root->tape_idx; i>=0; i--) {
        tape_memory[i].grad_fn(&tape_memory[i], tape_memory[i].prev);
    }

    if (!retain_graph) { // "default"
        free_vals();
    }
}

void update_params(float lr) {
    Value *v = parameters_head;
    while (v != NULL) {
        // gradient descent: data = data - (learning_rate * grad)
        v->data -= lr * v->grad;
        v = v->next;
    }
}

// the following functions are just for testing/analysis for Data Structures mini-project
// 
//
int testing() {
    return 0;
}

static void build_topo(Value *v, int *visited, Value **topo, int *topo_idx) {
    // If parameter or leaf node, skip (they recieve gradients from above)
    if (v->tape_idx < 0 || v->grad_fn == noop_backward) return;

    // If already visited (using unique tape_idx as the key), skip
    if (visited[v->tape_idx]) return;
    
    visited[v->tape_idx] = 1; // mark parent node
    
    // visit children first
    // note: in this case it's a bit weird - we are calling our previous inputs "children" 
    //       during our backwards pass, though in a forward pass they are the parents
    if (v->prev[0]) build_topo(v->prev[0], visited, topo, topo_idx);
    if (v->prev[1]) build_topo(v->prev[1], visited, topo, topo_idx);
    
    // post-order: add ourselves to the list _after_ children
    topo[*topo_idx] = v;
    (*topo_idx)++;
}

void backward_dfs(Value *root, bool retain_graph) {
    int *visited = calloc(tape_head, sizeof(int)); // calloc initializes to 0
    Value **topo = malloc(tape_head * sizeof(Value*)); // at most this many nodes to process
    int topo_idx = 0; // stores the 
    
    build_topo(root, visited, topo, &topo_idx);
    
    root->grad = 1.0;
    
    // iterate backwards from the end of the list
    // backpropagate in reverse topological order
    // (we built it children -> parent, so we iterate parent -> children)
    for (int i = topo_idx - 1; i >= 0; i--) {
        Value *v = topo[i];
        if (v->grad_fn) {
            v->grad_fn(v, v->prev);
        }
    }
    
    // cleanup
    free(visited);
    free(topo);
    if (!retain_graph) { // "default"
        free_vals();
    }
}
