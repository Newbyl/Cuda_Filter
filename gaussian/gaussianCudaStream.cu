#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

__constant__ double kernel[49] = {0, 0, 0, 5, 0, 0, 0,
                                  0, 5, 18, 32, 18, 5, 0,
                                  0, 18, 64, 100, 64, 18, 0,
                                  5, 32, 100, 100, 100, 32, 5,
                                  0, 18, 64, 100, 64, 18, 0,
                                  0, 5, 18, 32, 18, 5, 0,
                                  0, 0, 0, 5, 0, 0, 0};

// unsigned char *gaussian = matrice de sortie, g = matrice d'entree
__global__ void gaussian(unsigned char *gaussian, unsigned char *g, std::size_t cols, std::size_t rows)
{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;

  int tailleKernel = 7;

  if (x < cols && y < rows && x > 0 && y > 0)
  {

    double res = 0;
    double sommeKernel = 0;
    for (int j = -tailleKernel / 2; j <= tailleKernel / 2; ++j)
    {
      for (int i = -tailleKernel / 2; i <= tailleKernel / 2; ++i)
      {
        // coordonnées du pixel qui se fait filtrer
        int yy = y + j;
        int xx = x + i;
        if (yy < 0 || yy >= rows || xx < 0 || xx >= cols)
          continue;
        double weight = kernel[(j + tailleKernel / 2) * tailleKernel + i + tailleKernel / 2];
        res += weight * g[yy * cols + xx];
        sommeKernel += weight;
      }
    }

    gaussian[y * cols + x] = res / sommeKernel;
  }
}

int main(int argc, char **argv)
{
  cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  auto rows = m_in.rows;
  auto cols = m_in.cols;
  int nb_pixels = rows * cols;

  std::vector<unsigned char> g(rows * cols);

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  unsigned char *g_d;
  unsigned char *gaussian_d;
  unsigned char *pixels;
  unsigned char *res;

  cudaMallocHost(&pixels, nb_pixels);
  cudaMallocHost(&res, nb_pixels);

  pixels = m_in.data;

  cudaMalloc(&g_d, nb_pixels + cols);
  cudaMalloc(&gaussian_d, nb_pixels + cols);

  cudaStream_t streams[2];

  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  cudaMemcpyAsync(g_d, pixels, nb_pixels / 2 + cols, cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(g_d + nb_pixels / 2, pixels + (nb_pixels / 2) - cols, nb_pixels / 2 + cols, cudaMemcpyHostToDevice, streams[0]);

  dim3 t(32, 32);
  dim3 b((cols - 1) / t.x + 1, (rows - 1) / t.y + 1);

  cudaEventRecord(start, streams[0]);

  gaussian<<<b, t, 0, streams[0]>>>(gaussian_d, g_d, cols, (rows / 2) + 1);
  gaussian<<<b, t, 0, streams[1]>>>(gaussian_d + nb_pixels / 2, g_d + nb_pixels / 2 - cols, cols, (rows / 2) - 1);

  cudaEventRecord(stop, streams[0]);

  cudaMemcpyAsync(res, gaussian_d, nb_pixels / 2, cudaMemcpyDeviceToHost, streams[1]);
  cudaMemcpyAsync(res + nb_pixels / 2, gaussian_d + nb_pixels / 2 + cols, nb_pixels / 2, cudaMemcpyDeviceToHost, streams[1]);

  cudaDeviceSynchronize();

  cudaEventSynchronize(stop);

  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);

  cv::Mat m_out(rows, cols, CV_8UC1, res);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cv::imwrite("outStream.jpg", m_out);

  std::cout << ms << std::endl;

  cudaFree(g_d);
  cudaFree(gaussian_d);
  cudaFreeHost(pixels);
  cudaFreeHost(res);

  if (strcmp(cudaGetErrorString(cudaGetLastError()), "no error") != 0)
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  return 0;
}