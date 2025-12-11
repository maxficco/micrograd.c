CC = gcc
CFLAGS = -Wall -O2

all: test demo

micrograd.o: micrograd.c micrograd.h
	$(CC) $(CFLAGS) -c micrograd.c -o micrograd.o

neuralnetwork.o: neuralnetwork.c neuralnetwork.h micrograd.h
	$(CC) $(CFLAGS) -c neuralnetwork.c -o neuralnetwork.o

test: micrograd.o neuralnetwork.o test_micrograd.c
	$(CC) $(CFLAGS) -o test_suite test_micrograd.c micrograd.o neuralnetwork.o -lm
	./test_suite

demo: micrograd.o neuralnetwork.o demo_micrograd.c
	$(CC) $(CFLAGS) -o demo_run demo_micrograd.c micrograd.o neuralnetwork.o -lm
	./demo_run

clean:
	rm -f *.o test_suite demo_run
