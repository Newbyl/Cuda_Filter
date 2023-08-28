CXX = g++
CXXFLAGS = -O3 -march=native
LDLIBS= `pkg-config --libs opencv`

all: laplacian laplacianCuda laplacianCudaShared laplacianCudaStream 


laplacian: laplacian.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

laplacianCuda: laplacianCuda.cu
	nvcc -o $@ $< $(LDLIBS)

laplacianCudaShared: laplacianCudaShared.cu
	nvcc -o $@ $< $(LDLIBS)

laplacianCudaStream: laplacianCudaStream.cu
	nvcc -o $@ $< $(LDLIBS)

clean:
	rm laplacian
	rm laplacianCuda
	rm laplacianCudaShared
	rm laplacianCudaStream
