#include <stdio.h>
#include <omp.h>
#include <algorithm>
#include <limits.h>
#include <vector>

#define INFINITY INT_MAX

// Structure for vertex
typedef struct {
    int label;
    bool visited;
} Vertex;

// Structure for directed edge from u to v
typedef struct {
    int u;
    int v;
} Edge;

// Structure for storing path length and parent vertex
typedef struct {
    int length;
    int parent;
} PathInfo;

// Printing Shortest Path Length
void printShortestPathLength(PathInfo *path_info, int V) {
    printf("\nVERTEX \tSHORTEST PATH LENGTH \tPARENT\n");
    for (int i = 0; i < V; i++) {
        printf("%d \t", i);
        if (path_info[i].length < INFINITY)
            printf("%d \t\t\t%d\n", path_info[i].length, path_info[i].parent);
        else
            printf("Infinity \t\t\t-1\n");
    }
}

// Finds weight of the edge that connects Vertex u with Vertex v
int findEdgeWeight(int u, int v, Edge *edges, int *weights, int E) {
    for (int i = 0; i < E; i++) {
        if (edges[i].u == u && edges[i].v == v) {
            return weights[i];
        }
    }
    // If no edge exists, weight is infinity
    return INFINITY;
}

// Dijkstra Algorithm
void Dijkstra_Parallel(Vertex *vertices, Edge *edges, int *weights, Vertex *root, int V, int E) {
    double parallel_start, parallel_end;
    PathInfo *path_info = new PathInfo[V];

    // Initialize path lengths and parent vertices
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < V; i++) {
        path_info[i].length = INFINITY;
        path_info[i].parent = -1;
    }

    // Mark root vertex as visited, shortest path = 0
    root->visited = true;
    path_info[root->label].length = 0;
    path_info[root->label].parent = root->label;

    parallel_start = omp_get_wtime();
    // Parallelized Dijkstra's algorithm
    for (int iter = 0; iter < V; iter++) {
        int min_vertex = -1;
        int min_length = INFINITY;

        // Find the vertex with the shortest path among unvisited vertices
        #pragma omp parallel for schedule(static) shared(min_vertex, min_length)
        for (int i = 0; i < V; i++) {
            if (!vertices[i].visited && path_info[i].length < min_length) {
                #pragma omp critical
                {
                    if (path_info[i].length < min_length) {
                        min_length = path_info[i].length;
                        min_vertex = i;
                    }
                }
            }
        }

        // Mark the found vertex as visited
        vertices[min_vertex].visited = true;

        // Update shortest paths using the found vertex
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < V; i++) {
            if (!vertices[i].visited) {
                int weight = findEdgeWeight(vertices[min_vertex].label, vertices[i].label, edges, weights, E);
                if (weight != INFINITY) {
                    int new_length = min_length + weight;
                    #pragma omp critical
                    {
                        if (new_length < path_info[i].length) {
                            path_info[i].length = new_length;
                            path_info[i].parent = min_vertex;
                        }
                    }
                }
            }
        }
    }
    parallel_end = omp_get_wtime();

    printShortestPathLength(path_info, V);
    printf("\nRunning time: %lf ms\n", (parallel_end - parallel_start) * 1000);

    delete[] path_info;
}
int main() {
    int V, E;
    printf("Enter number of vertices: ");
    scanf("%d", &V);
    printf("Enter number of edges: ");
    scanf("%d", &E);

    // Allocate memory for vertices, edges, and weights
    Vertex *vertices = new Vertex[V];
    Edge *edges = new Edge[E];
    int *weights = new int[E];

    // Input edges and weights
    printf("\nEnter these details \nFROM \tTO \tWEIGHT\n");
    for (int i = 0; i < E; i++) {
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &weights[i]);
    }

    // Initialize vertices
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < V; i++) {
        vertices[i].label = i;
        vertices[i].visited = false;
    }

    int source;
    printf("\nEnter Source Vertex: ");
    scanf("%d", &source);

    Vertex root = {.label = source, .visited = false};
    Dijkstra_Parallel(vertices, edges, weights, &root, V, E);

    // Deallocate memory
    delete[] vertices;
    delete[] edges;
    delete[] weights;

    return 0;
}
