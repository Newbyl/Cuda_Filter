#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

// unsigned char *edge = matrice de sortie, g = matrice d'entree
__global__ void edge(unsigned char *edge, unsigned char *g, std::size_t cols, std::size_t rows)
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < cols + 1 && j < rows + 1 && i > 0 && j > 0)
  {
    // un pixel est calculé à partir de ses 8 voisins
    auto res = -1 * g[((j - 1) * cols + i - 1)] + -1 * g[((j - 0) * cols + i - 1)] + -1 * g[((j + 1) * cols + i - 1)] +
               -1 * g[((j - 1) * cols + i - 0)] + 8 * g[((j - 0) * cols + i + 0)] + -1 * g[((j + 1) * cols + i + 0)] +
               -1 * g[((j - 1) * cols + i + 1)] + -1 * g[((j - 0) * cols + i + 1)] + -1 * g[((j + 1) * cols + i + 1)];

    // on normalise le pixel (pas d'overshoot / undershoot )
    res = res > 255 ? res = 255 : res;
    res = res < 0 ? res = 0 : res;

    edge[j * cols + i] = res;
  }
}

int main(int argc, char **argv)
{
  // lecture du fichier avec openCV, on récupère une image en niveau de gris
  cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  auto rows = m_in.rows;
  auto cols = m_in.cols;
  int nb_pixels = rows * cols;

  // version applatie de l'image de sortie
  std::vector<unsigned char> g(rows * cols);

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  unsigned char *g_d;
  unsigned char *edge_d;
  unsigned char *pixels;
  unsigned char *res;

  // allocation de la mémoire "figée" sur l'hôte pour de meilleurs performances et obligatoire pour utiliser streams.
  cudaMallocHost(&pixels, nb_pixels);
  cudaMallocHost(&res, nb_pixels);

  pixels = m_in.data;

  // allocation de la mémoire sur le GPU
  cudaMalloc(&g_d, nb_pixels + cols);
  cudaMalloc(&edge_d, nb_pixels + cols);


  // création de 2 streams, 1 par moitié d'image
  cudaStream_t streams[2];

  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  // copie de l'image de l'hôte vers le GPU
  cudaMemcpyAsync(g_d, pixels, nb_pixels / 2 + cols, cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(g_d + nb_pixels / 2, pixels + (nb_pixels / 2) - cols, nb_pixels / 2 + cols, cudaMemcpyHostToDevice, streams[1]);

  dim3 t(32, 32);
  dim3 b((cols - 1) / t.x + 1, (rows - 1) / t.y + 1);

  cudaEventRecord(start, streams[0]);

  // lancement du kernel pour chaque moitié d'image
  edge<<<b, t, 0, streams[0]>>>(edge_d, g_d, cols, (rows / 2) + 1);
  edge<<<b, t, 0, streams[1]>>>(edge_d + nb_pixels / 2, g_d + nb_pixels / 2 - cols, cols, (rows / 2) - 1);

  cudaEventRecord(stop, streams[0]);

  // copie de l'image du GPU vers l'hôte
  cudaMemcpyAsync(res, edge_d, nb_pixels / 2, cudaMemcpyDeviceToHost, streams[0]);
  cudaMemcpyAsync(res + nb_pixels / 2, edge_d + nb_pixels / 2 + cols, nb_pixels / 2, cudaMemcpyDeviceToHost, streams[1]);

  // attend que chaque kernel terminent
  cudaDeviceSynchronize();

  cudaEventSynchronize(stop);

  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);

  // on crée une image avec opencv à partir du vecteur (CV_8UC1 : les pixels sont en 8bit et monocouleur)
  cv::Mat m_out(rows, cols, CV_8UC1, res);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cv::imwrite("outstream.jpg", m_out);

  std::cout << ms << std::endl;

  // libération de la mémoire
  cudaFree(g_d);
  cudaFree(edge_d);
  cudaFreeHost(pixels);
  cudaFreeHost(res);

  // gestion d'une éventuelle erreur
  if (strcmp(cudaGetErrorString(cudaGetLastError()), "no error") != 0)
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  return 0;
}
