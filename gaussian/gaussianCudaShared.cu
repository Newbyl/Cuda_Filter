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
__global__ void gaussianShared(unsigned char *gaussian, unsigned char *g, std::size_t cols, std::size_t rows)
{
    int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

    int shared_i = threadIdx.x;
    int shared_j = threadIdx.y;

    int w = blockDim.x;
    int h = blockDim.y;

    /* shared_g est la mémoire partagée et est initialisée à partir d'un paramètre donné au kernel avec le mot clé extern.
      Même si cela est plus souple que de hardcoder la taille, on ne peut pas créer de tableaux 2D.
    */
    extern __shared__ unsigned char shared_g[];

    // on remplie la mémoire partagée avec l'image donnée en entrée du kernel
    if (x < cols && y < rows)
    {
        shared_g[shared_j * w + shared_i] = g[y * cols + x];
    }

    // on s'assure que toute l'image a été copiée dans la mémoire partagée avant de la lire
    __syncthreads();

    int tailleKernel = 7;

    if (x < cols - 1 && y < rows - 1 && shared_i > 0 && shared_i < (w - 1) && shared_j > 0 && shared_j < (h - 1))
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
    unsigned char *gaussia_d;

    // allocation de la mémoire sur le GPU
    cudaMalloc(&g_d, rows * cols);
    cudaMalloc(&gaussia_d, rows * cols);

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
    gaussianShared<<<grid, block, block.x * block.y>>>(gaussia_d, g_d, cols, rows);

    cudaEventRecord(stop);

    cudaMemcpy(g.data(), gaussia_d, rows * cols, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cv::imwrite("outShared.jpg", m_out);

    std::cout << ms << std::endl;
    cudaFree(g_d);
    cudaFree(gaussia_d);

    // gestion d'une éventuelle erreur
    if (cudaGetLastError() != cudaSuccess)
        std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

    return 0;
}