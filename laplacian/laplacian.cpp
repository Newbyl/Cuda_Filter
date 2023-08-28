#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

int main(int argc, char **argv)
{
    cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    uchar *rgb = m_in.data;

    std::vector<unsigned char> g(m_in.rows * m_in.cols);
    cv::Mat m_out(m_in.rows, m_in.cols, CV_8UC1, g.data());

    auto start = std::chrono::system_clock::now();

    unsigned int i, j;

    int res;
    auto width = m_in.cols;
    auto height = m_in.rows;

    for (j = 1; j < height - 1; ++j)
    {
        for (i = 1; i < width - 1; ++i)
        {
            res = 0 * rgb[((j - 1) * width + i - 1)] + -1 * rgb[((j - 0) * width + i - 1)] + 0 * rgb[((j + 1) * width + i - 1)] +
                  -1 * rgb[((j - 1) * width + i - 0)] + 4 * rgb[((j - 0) * width + i + 0)] + -1 * rgb[((j + 1) * width + i + 0)] +
                  0 * rgb[((j - 1) * width + i + 1)] + -1 * rgb[((j - 0) * width + i + 1)] + 0 * rgb[((j + 1) * width + i + 1)];

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
