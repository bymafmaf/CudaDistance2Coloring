bfs: bfs.cu
	g++ graphio.c -c -O3
	g++ mmio.c -c -O3
	nvcc -O3 -c cudabfs.cu
	nvcc bfs.cu -c -O3
	g++ -o bfs bfs.o mmio.o graphio.o cudabfs.o -O3 -lcuda -lcudart -L/usr/local/cuda/lib64/ -fpermissive
clean:
	rm -f bfs *.o
