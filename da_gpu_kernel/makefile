default: lib/cudaDA.so
	python3 main.py
	
lib/cudaDA.so: cudaDigitalAnnealing.cu
	nvcc --compiler-options -fPIC -shared -arch sm_70 --maxrregcount=255 -o ./lib/cudaDA.so cudaDigitalAnnealing.cu

normal: cudaDigitalAnnealing.cu
	nvcc -arch sm_70 -o ./bin/cudaDA.o cudaDigitalAnnealing.cu
	./bin/cudaDA.o

debug: cudaDigitalAnnealing.cu
	nvcc -g -G -arch sm_70 -o coreDump cudaDigitalAnnealing.cu
	cuda-gdb ./coreDump
	rm ./coreDump

check: cudaDigitalAnnealing.cu
	nvcc --ptxas-options=-v -arch sm_70 --maxrregcount=255 -o coreDump cudaDigitalAnnealing.cu
	rm ./coreDump

runData: lib/cudaDA.so
	python3 toQUBO.py