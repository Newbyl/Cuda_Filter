CXX = g++
CXXFLAGS = -O3 -march=native
LDLIBS= `pkg-config --libs opencv`

all: edge edgeCuda edgeCudaShared edgeCudaStream 


edge: edge.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

edgeCuda: edgeCuda.cu
	nvcc -o $@ $< $(LDLIBS)

edgeCudaShared: edgeCudaShared.cu
	nvcc -o $@ $< $(LDLIBS)

edgeCudaStream: edgeCudaStream.cu
	nvcc -o $@ $< $(LDLIBS)

clean:
	rm edge
	rm edgeCuda
	rm edgeCudaShared
	rm edgeCudaStream
