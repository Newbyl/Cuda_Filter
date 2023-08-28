CXX = g++
CXXFLAGS = -O3 -march=native
LDLIBS= `pkg-config --libs opencv`

all: gaussian gaussianCuda gaussianCudaShared gaussianCudaStream 


gaussian: gaussian.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

gaussianCuda: gaussianCuda.cu
	nvcc -o $@ $< $(LDLIBS)

gaussianCudaShared: gaussianCudaShared.cu
	nvcc -o $@ $< $(LDLIBS)

gaussianCudaStream: gaussianCudaStream.cu
	nvcc -o $@ $< $(LDLIBS)

clean:
	rm gaussian
	rm gaussianCuda
	rm gaussianCudaShared
	rm gaussianCudaStream
