#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "micrograd.h"
#include "neuralnetwork.h"

// --- Helpers ---
int is_close(float a, float b) {
    return fabs(a - b) < 1e-4;
}

// --- Unit Tests ---

void test_basic_math() {
    printf("[TEST] Basic Math & Autograd... ");
    
    // z = a*b + c
    // a=2, b=3, c=1 -> z=7
    Value *a = new_val(2.0, NULL, NULL);
    Value *b = new_val(3.0, NULL, NULL);
    Value *c = new_val(1.0, NULL, NULL);
    Value *ab = mul(a, b);
    Value *z = add(ab, c);
    
    assert(is_close(z->data, 7.0));
    
    // Gradients:
    // dz/da = b = 3
    // dz/db = a = 2
    // dz/dc = 1
    zero_grad_all();
    backward(z, false);
    
    assert(is_close(a->grad, 3.0));
    assert(is_close(b->grad, 2.0));
    assert(is_close(c->grad, 1.0));
    
    printf("PASSED\n");
}

void test_activation() {
    printf("[TEST] ReLU & Tanh... ");
    
    // ReLU: f(-2)=0, f(5)=5
    Value *x1 = new_val(-2.0, NULL, NULL);
    Value *x2 = new_val(5.0, NULL, NULL);
    Value *r1 = relu(x1);
    Value *r2 = relu(x2);
    assert(r1->data == 0.0);
    assert(r2->data == 5.0);
    backward(r1, false);
    backward(r2, false);
    assert(is_close(x1->grad, 0.0));
    assert(is_close(x2->grad, 1.0));

    // Tanh: f(0)=0
    Value *t1 = v_tanh(new_val(0.0, NULL, NULL));
    assert(is_close(t1->data, 0.0));
    
    printf("PASSED\n");
    free_vals();
}

// --- Analysis & Benchmarks ---

void benchmark_model(int input_dim, int hidden_dim, int runs, char *label) {
    printf("[BENCHMARK] %s (Input: %d, Hidden: %d, Output: 10)\n", label, input_dim, hidden_dim);
    
    int nlayers = 3;
    int layerdims[] = {hidden_dim, hidden_dim, 10};
    MLP *mlp = new_mlp(input_dim, nlayers, layerdims);
    
    // Dummy inputs
    Value *x[input_dim];
    for(int i=0; i<input_dim; i++) x[i] = new_val(0.1f, NULL, NULL);
    
    clock_t start = clock();
    
    for(int i=0; i<runs; i++) {
        // Create fresh inputs on tape every step (simulating real training)
        for(int j=0; j<input_dim; j++) x[j] = new_val(0.1f, NULL, NULL);
        
        Value **out = forward(mlp, x);
        zero_grad();
        backward(out[0], false); // Linear sweep
    }
    
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    printf("   -> Time for %d passes: %.4f seconds\n", runs, cpu_time_used);

    free_mlp(mlp);
}

void compare_algorithms() {
    printf("\n=== ALGORITHM COMPARISON: Linear Sweep vs DFS ===\n");

    // ---------------------------------------------------------
    // TEST 1: The Disjoint Graph (Sparse)
    // Scenario: The tape is full of garbage nodes (noise) that 
    // contribute nothing to the final loss.
    // Prediction: DFS should win (skips garbage).
    // ---------------------------------------------------------
    printf("\n[CASE 1] Disjoint Graph (High Noise)\n");
    
    // 1. Create Noise (10,000 nodes disconnected from loss)
    for(int i=0; i<10000; i++) {
        Value *a = new_val(i, NULL, NULL);
        Value *b = new_val(i, NULL, NULL);
        mul(a, b);
    }
    
    // 2. Create Signal (500 nodes connected to loss)
    Value *head = new_val(1.0, NULL, NULL);
    for(int i=0; i<500; i++) {
        head = add(head, new_val(0.5, NULL, NULL));
    }
    Value *loss = head;

    // Benchmark Linear
    clock_t start = clock();
    for(int i=0; i<1000; i++) {
        zero_grad();
        backward(loss, true); // retain_graph=true to reuse data
    }
    double time_linear = ((double)(clock() - start)) / CLOCKS_PER_SEC;

    // Benchmark DFS
    start = clock();
    for(int i=0; i<1000; i++) {
        zero_grad();
        backward_dfs(loss, true);
    }
    double time_dfs = ((double)(clock() - start)) / CLOCKS_PER_SEC;

    printf("   Linear Time: %.4f s (Processed garbage)\n", time_linear);
    printf("   DFS Time:    %.4f s (Skipped garbage)\n", time_dfs);
    if (time_dfs < time_linear)
        printf("   >> WINNER: DFS (%.2fx faster)\n", time_linear/time_dfs);
    else
        printf("   >> WINNER: Linear\n");

    free_vals(); // RESET TAPE FOR ROUND 2

    // ---------------------------------------------------------
    // TEST 2: The Connected Graph (Dense)
    // Scenario: Every node on the tape is part of the computation.
    // Prediction: Linear Sweep should win (CPU Cache Locality).
    // ---------------------------------------------------------
    printf("\n[CASE 2] Fully Connected Graph (No Noise)\n");

    // 1. Create a massive chain (5000 nodes, all active)
    head = new_val(1.0, NULL, NULL);
    for(int i=0; i<5000; i++) {
        head = add(head, new_val(0.5, NULL, NULL));
    }
    loss = head;

    // Benchmark Linear
    start = clock();
    for(int i=0; i<1000; i++) {
        zero_grad();
        backward(loss, true);
    }
    time_linear = ((double)(clock() - start)) / CLOCKS_PER_SEC;

    // Benchmark DFS
    start = clock();
    for(int i=0; i<1000; i++) {
        zero_grad();
        backward_dfs(loss, true);
    }
    time_dfs = ((double)(clock() - start)) / CLOCKS_PER_SEC;

    printf("   Linear Time: %.4f s (Sequential RAM access)\n", time_linear);
    printf("   DFS Time:    %.4f s (Random pointer jumping)\n", time_dfs);
    
    if (time_linear < time_dfs)
        printf("   >> WINNER: Linear (%.2fx faster)\n", time_dfs/time_linear);
    else
        printf("   >> WINNER: DFS\n");
        
    free_vals(); // Cleanup
    printf("\n=============================================\n");
}

int main() {
    printf("=== MICROGRAD C TEST SUITE ===\n\n");
    
    test_basic_math();
    test_activation();
    
    printf("\n");
    benchmark_model(2, 4, 1000, "Small Model (XOR Size)");
    benchmark_model(64, 128, 1000, "Large Model");
    
    compare_algorithms();
    
    printf("\nAll tests completed successfully.\n");
    return 0;
}
