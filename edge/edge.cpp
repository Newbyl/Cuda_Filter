#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

int main(int argc, char **argv)
{
  // lecture du fichier avec openCV, on récupère une image en niveau de gris
  cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  auto rgb = m_in.data;

  // version applatie de l'image de sortie
  std::vector<unsigned char> g(m_in.rows * m_in.cols);
  cv::Mat m_out(m_in.rows, m_in.cols, CV_8UC1, g.data());

  // timing de l'application du filtre
  auto start = std::chrono::system_clock::now();

  unsigned int i, j;

  int res;
  auto width = m_in.cols;
  auto height = m_in.rows;

  for (j = 1; j < height - 1; ++j)
  {
    for (i = 1; i < width - 1; ++i)
    {
      // un pixel est calculé à partir de ses 8 voisins

      res = -1 * rgb[((j - 1) * width + i - 1)] + -1 * rgb[((j - 0) * width + i - 1)] + -1 * rgb[((j + 1) * width + i - 1)] +
            -1 * rgb[((j - 1) * width + i - 0)] + 8 * rgb[((j - 0) * width + i + 0)] + -1 * rgb[((j + 1) * width + i + 0)] +
            -1 * rgb[((j - 1) * width + i + 1)] + -1 * rgb[((j - 0) * width + i + 1)] + -1 * rgb[((j + 1) * width + i + 1)];

      // on normalise le pixel (pas d'overshoot / undershoot )
      res = res > 255 ? res = 255 : res;
      res = res < 0 ? res = 0 : res;

      g[j * width + i] = res;
    }
  }

  auto stop = std::chrono::system_clock::now();

  auto duration = stop - start;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  std::cout << ms << std::endl;

  cv::imwrite("outcpp.jpg", m_out);

  return 0;
}
