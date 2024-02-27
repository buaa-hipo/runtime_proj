#include <stdio.h>
#include <stdlib.h>
#include "mmio.h" // Include the mmio header file

typedef struct {
    int *row_ptr; // Pointer to row pointers
    int *col_idx; // Pointer to column indices
    double *values; // Pointer to non-zero values
    int rows; // Number of rows
    int cols; // Number of columns
    int nnz; // Number of non-zero elements
} CSRMatrixInfo;

inline CSRMatrixInfo* readCSRMatrix(const char* filename) {
    FILE *f;
    MM_typecode matcode;
    int m, n, nnz;
    int i, nz;
    int *I, *J;
    double *val;

    // Open the Matrix Market file
    if ((f = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Cannot open the file %s\n", filename);
        return NULL;
    }

    // Read matrix header
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        fclose(f);
        return NULL;
    }

    // Check matrix type
    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode) || !mm_is_real(matcode)) {
        fprintf(stderr, "Sorry, this function supports only real sparse matrices in Matrix Market format.\n");
        fclose(f);
        return NULL;
    }

    // Read matrix sizes
    if (mm_read_mtx_crd_size(f, &m, &n, &nnz) != 0) {
        fprintf(stderr, "Could not read matrix sizes.\n");
        fclose(f);
        return NULL;
    }

    // Allocate memory for matrix data
    I = (int *)malloc(nnz * sizeof(int));
    J = (int *)malloc(nnz * sizeof(int));
    val = (double *)malloc(nnz * sizeof(double));

    // Read matrix entries
    for (nz = 0; nz < nnz; nz++) {
        fscanf(f, "%d %d %lg\n", &I[nz], &J[nz], &val[nz]);
        I[nz]--; // Adjust to 0-based indexing
        J[nz]--;
    }

    // Close the file
    fclose(f);

    // Construct CSR matrix
    CSRMatrixInfo* csrMatrix = (CSRMatrixInfo*)malloc(sizeof(CSRMatrixInfo));
    csrMatrix->rows = m;
    csrMatrix->cols = n;
    csrMatrix->nnz = nnz;
    csrMatrix->row_ptr = (int *)calloc((m + 1), sizeof(int));
    csrMatrix->col_idx = (int *)malloc(nnz * sizeof(int));
    csrMatrix->values = (double *)malloc(nnz * sizeof(double));

    // Count non-zeros per row
    for (i = 0; i < nnz; i++) {
        csrMatrix->row_ptr[I[i] + 1]++;
    }

    // Cumulative sum to obtain row pointers
    for (i = 1; i <= m; i++) {
        csrMatrix->row_ptr[i] += csrMatrix->row_ptr[i - 1];
    }

    // Assign column indices and values
    for (i = 0; i < nnz; i++) {
        int row = I[i];
        int idx = csrMatrix->row_ptr[row];
        csrMatrix->col_idx[idx] = J[i];
        csrMatrix->values[idx] = val[i];
        csrMatrix->row_ptr[row]++;
    }

    // Shift row pointers back
    for (i = m; i > 0; i--) {
        csrMatrix->row_ptr[i] = csrMatrix->row_ptr[i - 1];
    }
    csrMatrix->row_ptr[0] = 0;

    // Return the CSR matrix info
    free(I);
    free(J);
    free(val);
    return csrMatrix;
}
