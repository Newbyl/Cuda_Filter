#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

// unsigned char *out_img = matrice de sortie, in_img = matrice d'entree
__global__ void kernel(unsigned char *out_img, unsigned char *in_img, std::size_t cols, std::size_t rows)
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < cols && j < rows && i > 1 && j > 1)
  {
    // un pixel est calculé à partir de ses 8 voisins
    auto pixel = 0 * in_img[((j - 1) * cols + i - 1)] + -1 * in_img[((j - 0) * cols + i - 1)] + 0 * in_img[((j + 1) * cols + i - 1)] +
                 -1 * in_img[((j - 1) * cols + i - 0)] + 4 * in_img[((j - 0) * cols + i + 0)] + -1 * in_img[((j + 1) * cols + i + 0)] +
                 0 * in_img[((j - 1) * cols + i + 1)] + -1 * in_img[((j - 0) * cols + i + 1)] + 0 * in_img[((j + 1) * cols + i + 1)];

    // on normalise le pixel (pas d'overshoot / undershoot )
    pixel = pixel > 255 ? pixel = 255 : pixel;
    pixel = pixel < 0 ? pixel = 0 : pixel;

    out_img[j * cols + i] = pixel;
  }
}

int main(int argc, char **argv)
{
  // lecture du fichier avec openCV, on récupère une image en niveau de gris
  cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  auto pixels = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;

  // version applatie de l'image de sortie
  std::vector<unsigned char> new_img(rows * cols);

  // on crée une image avec opencv à partir du vecteur (CV_8UC1 : les pixels sont en 8bit et monocouleur)
  cv::Mat m_out(rows, cols, CV_8UC1, new_img.data());

  cudaEvent_t start, stop;

  // pour le monitoring du temps mis par le kernel
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  unsigned char *img_in_d;
  unsigned char *img_out_d;

  // allocation de la mémoire sur le GPU
  cudaMalloc(&img_in_d, rows * cols);
  cudaMalloc(&img_out_d, rows * cols);

  // copie de l'image de l'hôte vers le GPU
  cudaMemcpy(img_in_d, pixels, rows * cols, cudaMemcpyHostToDevice);

  dim3 t(32, 32);
  dim3 b((cols - 1) / t.x + 1, (rows - 1) / t.y + 1);

  cudaEventRecord(start);

  // lancement du kernel
  kernel<<<b, t>>>(img_out_d, img_in_d, cols, rows);

  cudaEventRecord(stop);

  // copie de l'image du GPU vers l'hôte
  cudaMemcpy(new_img.data(), img_out_d, rows * cols, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cv::imwrite("outcuda.jpg", m_out);

  std::cout << ms << std::endl;

  cudaFree(img_in_d);
  // gestion d'une éventuelle erreur
  if (cudaGetLastError() != cudaSuccess)
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 0;
}
