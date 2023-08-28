#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

__global__ void line(unsigned char *out, unsigned char *in, std::size_t cols, std::size_t rows)
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < cols && j < rows && i > 1 && j > 1)
  {
    // un pixel est calculé à partir de ses 8 voisins

    auto res = -1 * in[((j - 1) * cols + i - 1)] + -1 * in[((j - 0) * cols + i - 1)] + 2 * in[((j + 1) * cols + i - 1)] +
               -1 * in[((j - 1) * cols + i - 0)] + 2 * in[((j - 0) * cols + i + 0)] + -1 * in[((j + 1) * cols + i + 0)] +
               2 * in[((j - 1) * cols + i + 1)] + -1 * in[((j - 0) * cols + i + 1)] + -1 * in[((j + 1) * cols + i + 1)];

    // on normalise le pixel (pas d'overshoot / undershoot )

    res = res > 255 ? res = 255 : res;
    res = res < 0 ? res = 0 : res;

    out[j * cols + i] = res;
  }
}

int main(int argc, char **argv)
{
    // lecture du fichier avec openCV, on récupère une image en niveau de gris

  cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

    // version applatie de l'image de sortie

  std::vector<unsigned char> g(rows * cols);
  cv::Mat m_out(rows, cols, CV_8UC1, g.data());

  cudaEvent_t start, stop;

  // on crée une image avec opencv à partir du vecteur (CV_8UC1 : les pixels sont en 8bit et monocouleur)

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  unsigned char *g_d;
  unsigned char *line_d;

  // allocation de la mémoire sur le GPU
  cudaMalloc(&g_d, rows * cols);
  cudaMalloc(&line_d, rows * cols);

  // copie de l'image de l'hôte vers le GPU
  cudaMemcpy(g_d, rgb, rows * cols, cudaMemcpyHostToDevice);

  dim3 t(32, 32);
  dim3 b((cols - 1) / t.x + 1, (rows - 1) / t.y + 1);

  cudaEventRecord(start);

  // lancement du kernel
  line<<<b, t>>>(line_d, g_d, cols, rows);

  cudaEventRecord(stop);

  // copie de l'image du GPU vers l'hôte
  cudaMemcpy(g.data(), line_d, rows * cols, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cv::imwrite("outcuda.jpg", m_out);
  cudaFree(g_d);

  // gestion d'une éventuelle erreur
  if (cudaGetLastError() != cudaSuccess)
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  return 0;
}