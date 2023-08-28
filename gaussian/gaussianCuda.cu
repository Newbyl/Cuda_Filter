#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

// matrice de convolution stockée dans la mémoire cache du gpu
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

  // Largeur du kernel pour les calculs
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
  // lecture du fichier avec openCV, on récupère une image en niveau de gris
  cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  // version applatie de l'image de sortie
  std::vector<unsigned char> g(rows * cols);

  // on crée une image avec opencv à partir du vecteur (CV_8UC1 : les pixels sont en 8bit et monocouleur)
  cv::Mat m_out(rows, cols, CV_8UC1, g.data());

  unsigned char *g_d;
  unsigned char *gaussian_d;

  // allocation de la mémoire sur le GPU
  cudaMalloc(&g_d, rows * cols);
  cudaMalloc(&gaussian_d, rows * cols);

  // copie de l'image de l'hôte vers le GPU

  cudaMemcpy(g_d, m_in.data, rows * cols, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;

  // pour le monitoring du temps mis par le kernel

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 block(32, 32);
  dim3 grid((cols - 1) / (block.x - 2) + 1, (rows - 1) / (block.y - 2) + 1);

  cudaEventRecord(start);

  // lancement du kernel
  gaussian<<<grid, block, block.x * block.y>>>(gaussian_d, g_d, cols, rows);

  cudaEventRecord(stop);

  // copie de l'image du GPU vers l'hôte
  cudaMemcpy(g.data(), gaussian_d, rows * cols, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cv::imwrite("outcuda.jpg", m_out);

  std::cout << ms << std::endl;
  cudaFree(g_d);
  cudaFree(gaussian_d);

  // gestion d'une éventuelle erreur
  if (cudaGetLastError() != cudaSuccess)
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  return 0;
}
