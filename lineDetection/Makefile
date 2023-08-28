CXX = g++
CXXFLAGS = -O3 -march=native
LDLIBS= `pkg-config --libs opencv`

all: line lineCuda lineCudaShared lineCudaStream


line: line.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

lineCuda: linecuda.cu
	nvcc -o $@ $< $(LDLIBS)

lineCudaShared: lineCudaShared.cu
	nvcc -o $@ $< $(LDLIBS)

lineCudaStream: lineCudaStream.cu
	nvcc -o $@ $< $(LDLIBS)

clean:
	rm line
	rm lineCuda
	rm lineCudaShared
	rm lineCudaStream
