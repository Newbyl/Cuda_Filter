#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <cmath>

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

    // la matrice de convolution
    double kernel[49] = {0, 0, 0, 5, 0, 0, 0,
                         0, 5, 18, 32, 18, 5, 0,
                         0, 18, 64, 100, 64, 18, 0,
                         5, 32, 100, 100, 100, 32, 5,
                         0, 18, 64, 100, 64, 18, 0,
                         0, 5, 18, 32, 18, 5, 0,
                         0, 0, 0, 5, 0, 0, 0};
    int tailleKernel = 7;

    for (int y = 0; y < m_in.rows; ++y)
    {
        for (int x = 0; x < m_in.cols; ++x)
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
                    if (yy < 0 || yy >= m_in.rows || xx < 0 || xx >= m_in.cols)
                        continue;
                    double weight = kernel[(j + tailleKernel / 2) * tailleKernel + i + tailleKernel / 2];
                    res += weight * rgb[yy * m_in.cols + xx];
                    sommeKernel += weight;
                }
            }
            g[y * m_in.cols + x] = res / sommeKernel;
        }
    }

    auto stop = std::chrono::system_clock::now();

    auto duration = stop - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::cout << ms << std::endl;

    cv::imwrite("outcpp.jpg", m_out);

    return 0;
}
