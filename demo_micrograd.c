#include <stdio.h>
#include <time.h>
#include "micrograd.h"
#include "neuralnetwork.h"


void demo_calculus() {
    printf("\n--- 1. Intuitive Demo: Multivariable Calculus ---\n");
    printf("Equation: f(a, b) = a^2 + 3b - 5\n");
    printf("We want to find how 'f' changes as we tweak 'a' and 'b'.\n\n");

    // 1. Define Inputs
    // Let a = 3.0, b = 2.0
    Value *a = new_val(3.0, NULL, NULL);
    Value *b = new_val(2.0, NULL, NULL);
    
    // 2. Build the Graph (Forward Pass)
    // f = a^2 + 3*b - 5
    Value *a_squared = v_pow(a, 2);           // a^2 = 9
    Value *three_b = mul(new_val(3, NULL, NULL), b); // 3*b = 6
    Value *sum = add(a_squared, three_b);          // 9 + 6 = 15
    Value *f = add(sum, new_val(-5, NULL, NULL)); // 15 - 5 = 10
    
    printf("Forward Pass Results:\n");
    printf("   a = %.2f\n", a->data);
    printf("   b = %.2f\n", b->data);
    printf("   f = %.2f (Expected: 3^2 + 3*2 - 5 = 10)\n", f->data);
    
    // 3. Backward Pass
    backward(f, false); // false = clear graph after
    
    printf("\nBackward Pass (Gradients):\n");
    
    // df/da = 2a = 2(3) = 6
    printf("   df/da: %.2f (Expected: 2*a = 6.0)\n", a->grad);
    
    // df/db = 3
    printf("   df/db: %.2f (Expected: Constant slope 3.0)\n", b->grad);
    
    printf("\n[Intuition]: If we nudge 'a' up by 0.01, 'f' will grow by ~0.06.\n");
    printf("-------------------------------------------------\n");
    free_vals();
}

void demo_neuron() {
    printf("\n--- 2. Intuitive Demo: A Single Neuron ---\n");
    printf("Equation: output = tanh(w * x + bias)\n");
    printf("This is the fundamental atom of Deep Learning.\n\n");
    
    // 1. Inputs
    Value *x = new_val(1.0, NULL, NULL);  // Input
    Value *w = new_val(0.5, NULL, NULL);  // Weight
    Value *b = new_val(0.2, NULL, NULL);  // Bias
    
    // 2. Operations (Forward Pass)
    Value *wx = mul(w, x);            // 0.5 * 1.0 = 0.5
    Value *z = add(wx, b);            // 0.5 + 0.2 = 0.7
    Value *out = v_tanh(z);           // tanh(0.7) ≈ 0.604
    
    printf("Forward Pass:\n");
    printf("   Input (x):  %.2f\n", x->data);
    printf("   Weight (w): %.2f\n", w->data);
    printf("   Bias (b):   %.2f\n", b->data);
    printf("   Result:     %.4f\n", out->data);
    
    // 3. Backward Pass
    backward(out, false);
    
    // 4. Results
    printf("\nBackward Pass (Sensitivity):\n");
    printf("   d(out)/d(w): %.4f (How much the weight matters)\n", w->grad);
    printf("   d(out)/d(x): %.4f (How much the input matters)\n", x->grad);
    
    printf("-------------------------------------------------\n");
    free_vals();
}

void demo_xor() {
    printf("\n--- 3. Training Demo: Solving XOR ---\n");
    printf("Training a 2-layer MLP to solve the XOR problem.\n");
    
    // define XOR dataset
    int inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int targets[4] = {0, 1, 1, 0};

    // define multi-layer perceptron
    int inputdim = 2;
    int nlayers = 2;
    int layerdims[2] = {4, 1}; //hidden=4, output=1
    MLP *mlp = new_mlp(inputdim, nlayers, layerdims);
    
    printf("Model initialized. Training for 10000 steps...\n");
    // training loop
    for (int step=0; step<10000; step++) {
        Value *total_loss = new_val(0, NULL, NULL);
        // batch loop
        for (int i=0; i<4; i++) {
            Value *x[2];
            x[0] = new_val(inputs[i][0], NULL, NULL);
            x[1] = new_val(inputs[i][1], NULL, NULL);

            Value *y = new_val(targets[i], NULL, NULL);

            Value **out = forward(mlp, x);

            Value *diff = sub(out[0], y);
            Value *mse = v_pow(diff, 2);
            total_loss = add(total_loss, mse);
        }
        zero_grad();
        backward(total_loss, false);
        update_params(0.005);

        if (step%500 == 0) {
            printf("Step: %-4d | Loss: %.8f\n", step, total_loss->data);
        }
    }

    // check results
    printf("Results:\n");
    for (int i=0; i<4; i++) {
        Value *x[2];
        x[0] = new_val(inputs[i][0], NULL, NULL);
        x[1] = new_val(inputs[i][1], NULL, NULL);

        Value **out = forward(mlp, x);

        printf("%d ^ %d = %f (target: %d)\n", inputs[i][0], inputs[i][1], out[0]->data, targets[i]);
    }

    free_mlp(mlp);
}

int main() {
    srand(time(NULL));
    demo_calculus();
    demo_neuron();
    demo_xor();
    return 0;
}
