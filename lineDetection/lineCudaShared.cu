#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

__global__ void line(unsigned char *out, unsigned char *in, std::size_t cols, std::size_t rows)
{
    int i = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    int j = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

    int shared_i = threadIdx.x;
    int shared_j = threadIdx.y;

    int w = blockDim.x;
    int h = blockDim.y;

    /* shared_g est la mémoire partagée et est initialisée à partir d'un paramètre donné au kernel avec le mot clé extern.
      Même si cela est plus souple que de hardcoder la taille, on ne peut pas créer de tableaux 2D.
    */
    extern __shared__ unsigned char shared_g[];

    // on remplie la mémoire partagée avec l'image donnée en entrée du kernel
    if (i < cols && j < rows)
    {
        shared_g[shared_j * w + shared_i] = in[j * cols + i];
    }

    // on s'assure que toute l'image a été copiée dans la mémoire partagée avant de la lire
    __syncthreads();

    if (i < cols - 1 && j < rows - 1 && shared_i > 0 && shared_i < (w - 1) && shared_j > 0 && shared_j < (h - 1))
    {
        // un pixel est calculé à partir de ses 8 voisins
        auto res = -1 * in[((j - 1) * cols + i - 1)] + -1 * in[((j - 0) * cols + i - 1)] + 2 * in[((j + 1) * cols + i - 1)] +
                   -1 * in[((j - 1) * cols + i - 0)] + 2 * in[((j - 0) * cols + i + 0)] + -1 * in[((j + 1) * cols + i + 0)] +
                   2 * in[((j - 1) * cols + i + 1)] + -1 * in[((j - 0) * cols + i + 1)] + -1 * in[((j + 1) * cols + i + 1)];

        res = min(255, max(0, res));
        out[j * cols + i] = res;
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
    unsigned char *line_d;

    // allocation de la mémoire sur le GPU
    cudaMalloc(&g_d, rows * cols);
    cudaMalloc(&line_d, rows * cols);

    // copie de l'image de l'hôte vers le GPU
    cudaMemcpy(g_d, m_in.data, rows * cols, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;

    // pour le monitoring du temps mis par le kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 block(32, 4);

    dim3 grid((cols - 1) / (block.x - 2) + 1, (rows - 1) / (block.y - 2) + 1);

    cudaEventRecord(start);

    // lancement du kernel
    line<<<grid, block, block.x * block.y>>>(line_d, g_d, cols, rows);

    cudaEventRecord(stop);

    cudaMemcpy(g.data(), line_d, rows * cols, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cv::imwrite("outshared.jpg", m_out);

    std::cout << ms << std::endl;
    cudaFree(g_d);
    cudaFree(line_d);

    // gestion d'une éventuelle erreur
    if (cudaGetLastError() != cudaSuccess)
        std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

    return 0;
}
