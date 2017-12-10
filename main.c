#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 31

typedef struct {
    int left;
    int right;
} Edges;

typedef struct {
    int start;
    int finish;
    int rank;
} RectSide;

int count_non_white_pixels(int *picture_partial, Edges *row_edges, RectSide *side)
{
    int non_white_count = 0;
    row_edges->left = N;
    row_edges->right = -1;
    side->start = -1;
    side->finish = -1;
    for (int j=1; j<N; j++)
    {
        if (picture_partial[j] != 0)
        {
//            printf("Process %d found a non-white spot.\n", side->rank);
            non_white_count++;
            if (j < row_edges->left)
            {
//                printf("Process %d found set the left edge to %d.\n", side->rank, j);
                row_edges->left = j;
            }
            if (j > row_edges->right)
            {
//                printf("Process %d found set the right edge to %d.\n", side->rank, j);
                row_edges->right = j;
            }
            if (j != N-1 && picture_partial[j+1] != 0)
            {
                if (side->start < 0)
                {
                    side->start = j;
                }
                side->finish = j+1;
            }
        }
    }

    return non_white_count;
}

int* to_flat_array(int pic[N][N])
{
    int *result = (int *) malloc(sizeof(int)*N*N);
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            result[i*N+j] = pic[i][j];
        }
    }

    return result;
}

void print_total_non_white_pixels(int sub_non_white_count, int rank)
{
    int total_non_white;
    MPI_Reduce(&sub_non_white_count, &total_non_white, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("The number of non-white pixels of the given image is %d\n", total_non_white);
    }
}

void print_x_side_length(Edges row_edges, int rank)
{
    int left = 0;
    MPI_Reduce(&row_edges.left, &left, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    int right = 0;
    MPI_Reduce(&row_edges.right, &right, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("The x side of the rectangle with minimal face that contains the image is %d\n", right - left + 1);
    }
}

int compare_int(const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b );
}

int compare_sides(const void * a, const void * b)
{
    RectSide *sidesA = (RectSide *)a;
    RectSide *sidesB = (RectSide *)b;

    return (sidesA->rank - sidesB->rank);
}

void print_y_side_length(int rank, Edges row_edges)
{
    int nonzero = -1;
    if (row_edges.left != N && row_edges.right != -1)
    {
        nonzero = rank;
    }

    int *non_zero_rows = NULL;
    if (rank == 0) {
      non_zero_rows = (int *)malloc(sizeof(int) * N);
    }
    MPI_Gather(&nonzero, 1, MPI_INT, non_zero_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        qsort(non_zero_rows, N, sizeof(int), compare_int);
        int min = N;
        int max = -1;
        for (int i=0; i<N; i++) {
            if (non_zero_rows[i] >= 0)
            {
                int current_rank = non_zero_rows[i];
                if (min > current_rank) {
                    min = current_rank;
                }
                if (max < current_rank) {
                    max = current_rank;
                }
            }
        }
        printf("The y side of the rectangle with minimal face that contains the image is %d\n", max - min + 1);
    }
}

void print_rect_coords(RectSide side, int rank)
{
    RectSide *sides = NULL;
    if (rank == 0) {
        sides = (RectSide *)malloc(sizeof(RectSide) * N);
    }
    MPI_Gather(&side, sizeof(RectSide), MPI_BYTE, sides, sizeof(RectSide), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        qsort(sides, N, sizeof(RectSide), compare_sides);

        int left = sides[1].start;
        int right = sides[1].finish;
        int height = 1;
        for (int i=1; i<N-1; i++)
        {
            if (left >= sides[i+1].finish || right <= sides[i+1].start)
            {
                if (height > 1)
                {
                    printf("Top left: (%d,%d), Bottom right: (%d,%d)\n", (i-height+1), left, i, right);
                }
                left = sides[i+1].start;
                right = sides[i+1].finish;
                height = 1;
            } else
            {
                left = sides[i].start > sides[i+1].start ? sides[i].start : sides[i+1].start;
                right = sides[i].finish < sides[i+1].finish ? sides[i].finish : sides[i+1].finish;
                height++;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (N != world_size)
    {
        fprintf(stderr, "The number of processes should be equal to %d\n", N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* create a type for struct RectSide */
    const int nitems=3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Datatype mpi_rectside_type;
    MPI_Aint offsets[3];

    offsets[0] = offsetof(RectSide, start);
    offsets[1] = offsetof(RectSide, finish);
    offsets[0] = offsetof(RectSide, rank);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_rectside_type);
    MPI_Type_commit(&mpi_rectside_type);

    int *flat_pic = NULL;
    if (rank == 0)
    {
        int pic[N][N] =   { {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                          {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
                          };

                /*{
            {0,0,0,0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0,0,0,0},{0,0,1,1,1,0,0,0,0,0,0},
            {0,0,0,1,1,0,0,0,0,0,0},{0,0,0,0,1,0,1,0,1,0,0},
            {0,0,0,0,1,1,1,1,1,1,0},{0,0,0,0,0,1,1,1,1,0,0},
            {0,0,0,0,0,1,1,1,1,0,0},{0,0,0,0,0,1,0,0,1,0,0},
            {0,0,0,0,0,1,0,0,1,0,0},{0,0,0,0,0,1,0,0,1,0,0}
        };*/

        flat_pic = to_flat_array(pic);
    }

    int *picture_partial = (int *)malloc(sizeof(int) * N);
    MPI_Scatter(flat_pic, N, MPI_INT, picture_partial, N, MPI_INT, 0, MPI_COMM_WORLD);

    Edges row_edges;
    RectSide rectside;
    rectside.rank = rank;
    int sub_non_white_count = count_non_white_pixels(picture_partial, &row_edges, &rectside);

    print_total_non_white_pixels(sub_non_white_count, rank);
    print_x_side_length(row_edges, rank);
    print_y_side_length(rank, row_edges);
    print_rect_coords(rectside, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
