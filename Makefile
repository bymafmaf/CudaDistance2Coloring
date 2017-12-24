bfs: bfs.cu
	g++ graphio.c -c -O3
	g++ mmio.c -c -O3
	nvcc -O3 -c cudabfs.cu
	nvcc bfs.cu -c -O3 -std=c++11
	g++ -o bfs bfs.o mmio.o graphio.o cudabfs.o -O3 -lcuda -lcudart -L/usr/local/cuda/lib64/ -fpermissive

omp: omp.cpp
	gcc graphio.c -c -O3
	gcc mmio.c -c -O3
	g++ -O3 -c omp.cpp -std=c++11 -fopenmp
	gcc -o omp omp.o mmio.o graphio.o -fopenmp -lstdc++ -O3
clean:
	rm -f bfs *.o
