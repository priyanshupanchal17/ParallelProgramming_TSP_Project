#include<stdio.h>
#include<omp.h>
#include<time.h>
#include<unistd.h>
#include<limits.h>

#define VERTICES 1000

int main()
{
    clock_t begin = clock();

    int n;
    printf("Enter No Of Vertices : ");
    scanf("%d", &n);
    if (n <= 0 || n > VERTICES) {
        printf("Invalid number of vertices\n");
        return 1;
    }

    int a[n][n]; // Adjacency matrix
    int mstSet[n]; // Set to keep track of vertices included in MST
    int key[n]; // Key values used to pick the minimum weight edge
    int parent[n]; // Array to store the constructed MST
    int i, j;

    // Initialize all keys as infinite
    for (i = 0; i < n; i++) {
        key[i] = INT_MAX;
        mstSet[i] = 0;
    }

    // Input graph
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("Enter Edge Weight (%d, %d) : ", i, j);
            scanf("%d", &a[i][j]);
        }
    }

    // Always include first vertex in MST
    key[0] = 0;
    parent[0] = -1;

    // Find MST using Prim's algorithm
    for (i = 0; i < n - 1; i++) {
        // Find vertex with minimum key value
        int minKey = INT_MAX;
        int minIndex;
        #pragma omp parallel for shared(minKey, minIndex) // Parallelized loop to find minimum key value
        for (j = 0; j < n; j++) {
            if (mstSet[j] == 0 && key[j] < minKey) {
                #pragma omp critical
                {
                    if (key[j] < minKey) {
                        minKey = key[j];
                        minIndex = j;
                    }
                }
            }
        }

        // Add the picked vertex to MST Set
        mstSet[minIndex] = 1;

        // Update key values and parent index of the adjacent vertices
        #pragma omp parallel for
        for (j = 0; j < n; j++) {
            if (a[minIndex][j] && mstSet[j] == 0 && a[minIndex][j] < key[j]) {
                key[j] = a[minIndex][j];
                parent[j] = minIndex;
            }
        }
    }

    // Print MST and calculate total weight
    int totalWeight = 0;
    printf("Edge   Weight\n");
    for (i = 1; i < n; i++) {
        printf("%d - %d    %d \n", parent[i], i, a[i][parent[i]]);
        totalWeight += a[i][parent[i]];
    }

    printf("Total Weight of MST: %d\n", totalWeight);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent: %lf\n", time_spent);

    return 0;
}
