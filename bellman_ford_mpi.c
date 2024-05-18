#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector> // Added for using vectors
using namespace std;
#include "mpi.h"

using std::cout;
using std::endl;
using std::string;

#define INF 1000000

namespace utils {
    int N;  // number of vertices
    int* mat; // the adjacency matrix

    void abort_with_error_message(string msg) {
        std::cerr << msg << endl;
        abort();
    }

    int convert_dimension_2D_1D(int x, int y, int n) {
        return x * n + y;
    }

    void read_input(int my_rank) {
        if (my_rank == 0) {
            cout << "Enter the number of vertices: ";
            cin >> N;
            mat = new int[N * N];
            cout << "Enter the adjacency matrix (" << N << "x" << N << "):" << endl;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    cin >> mat[convert_dimension_2D_1D(i, j, N)];
                }
        }
    }

    std::pair<int, std::vector<int>> print_result(bool has_negative_cycle, int* dist) {
        int path_length = 0;
        std::vector<int> path;
        if (!has_negative_cycle) {
            for (int i = 0; i < N; i++) {
                if (dist[i] > INF)
                    dist[i] = INF;
                path_length += dist[i];
                path.push_back(dist[i]); // Store the path
            }
        }
        cout << "Path: ";
        for (int node: path) {
            cout << node << " ";
        }
        cout << endl;
        cout << "Path Cost: " << (has_negative_cycle ? "FOUND NEGATIVE CYCLE!" : std::to_string(path_length)) << endl;
        return { has_negative_cycle ? -1 : path_length, path };
    }
}

void bellman_ford(int my_rank, int p, MPI_Comm comm, int n, int* mat, int* dist, bool* has_negative_cycle)
{
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);

    int* my_dist;

    int* load = new int[p];
    int* begin = new int[p];
    load[0] = n / p;
    for (int i = 1; i < p; ++i)
        load[i] = n / p + ((i < n % p) ? 1 : 0);
    begin[0] = 0;
    for (int i = 1; i < p; ++i)
        begin[i] = begin[i - 1] + load[i - 1];

    int my_load = load[my_rank];
    int my_begin = begin[my_rank];
    int my_end = my_begin + my_load;

    my_dist = new int[n];

    if (my_rank != 0)
        mat = new int[n * n];
    MPI_Bcast(mat, n * n, MPI_INT, 0, comm);

    for (int i = 0; i < n; i++)
        my_dist[i] = INF;

    my_dist[0] = 0;
    MPI_Barrier(comm);

    bool my_has_change;
    int my_iter_num = 0;
    for (int i = 0; i < n - 1; i++) {
        my_has_change = false;
        my_iter_num++;
        for (int u = 0; u < n; u++) {
            if (my_dist[u] == INF)
                continue;
            for (int v = my_begin; v < my_end; v++)
            {
                int weight = mat[u * n + v];
                if (weight < INF) {
                    if (my_dist[u] + weight < my_dist[v]) {
                        my_dist[v] = my_dist[u] + weight;
                        my_has_change = true;
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &my_has_change, 1, MPI_C_BOOL, MPI_LOR, comm);
        if (!my_has_change)
            break;
        MPI_Allreduce(MPI_IN_PLACE, my_dist, n, MPI_INT, MPI_MIN, comm);
    }

    if (my_iter_num == n - 1) {
        my_has_change = false;
        for (int u = 0; u < n; u++) {
            for (int v = my_begin; v < my_end; v++) {
                int weight = mat[u * n + v];
                if (weight < INF && my_dist[u] + weight < my_dist[v]) {
                    my_has_change = true;
                    break;
                }
            }
            if (my_has_change)
                break;
        }
        MPI_Allreduce(&my_has_change, has_negative_cycle, 1, MPI_C_BOOL, MPI_LOR, comm);
    }

    if (my_rank == 0)
        memcpy(dist, my_dist, n * sizeof(int));

    if (my_rank != 0)
        delete[] mat;
    delete[] my_dist;
    delete[] load;
    delete[] begin;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm;

    int p;
    int my_rank;
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    utils::read_input(my_rank);

    int* dist = new int[utils::N];
    bool has_negative_cycle = false;

    double t1, t2;
    MPI_Barrier(comm);
    t1 = MPI_Wtime();

    bellman_ford(my_rank, p, comm, utils::N, utils::mat, dist, &has_negative_cycle);
    MPI_Barrier(comm);
    t2 = MPI_Wtime();

    if (my_rank == 0) {
        std::cerr.setf(std::ios::fixed);
        std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
        auto result = utils::print_result(has_negative_cycle, dist);
        delete[] dist;
        delete[] utils::mat;
    }

    MPI_Finalize();
    return 0;
}
