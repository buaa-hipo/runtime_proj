#include<omp.h>
#include<cmath>
#include<vector>
#include<array>
#include<iostream>
#include<chrono>

#include "mm_highlevel.h"

#include "mkl.h"

bool ignore_ksp_stop = false;

using std::vector;

using ElemType=double;

struct ParaVector {
    std::vector<ElemType> vec;
    int nelems;
    explicit ParaVector(int n) : vec(n), nelems(n) {}

    ElemType serial_partial_vdot(const ParaVector& v1, int start, int end);
    ElemType parallel_vdot(const ParaVector& v1, int ntasks);
    void axpy(const ParaVector& v1, ElemType a, int ntasks);
    void serial_partial_axpy(const ParaVector& v1, ElemType a, int start, int len);
    void scale(ElemType a, int ntasks);
    void serial_partial_scale(ElemType a, int start, int len);
    ElemType norm2(int ntasks);
    void set_val(ElemType val);
};

void ParaVector::set_val(ElemType val) {
    for (int i = 0; i < nelems; i++) {
        vec[i] = val;
    }
}

ElemType ParaVector::serial_partial_vdot(const ParaVector& v1, int start, int len) {
    // printf("vdot: %d %d\n", start, len);
    if (std::is_same<ElemType, double>::value) {
        static int incy = 1;
        ElemType res = ddot(&len, vec.data() + start, &incy, v1.vec.data() + start, &incy);
        return res;
    }
    exit(-1);
}

ElemType ParaVector::parallel_vdot(const ParaVector& v1, int ntasks) {
    if (v1.nelems != nelems) {
        printf("Got vdot for vectors of different lengths!\n");
        exit(-1);
    }

    int per_task_elems = floor(1.0 * nelems / ntasks);

    ElemType sum = 0;

    #pragma omp taskgroup task_reduction(+:sum)
    {
        for (int i = 0; i < ntasks; i++) {
            #pragma omp task in_reduction(+:sum) shared(vec, v1, per_task_elems, nelems, ntasks)
            {
                int start = per_task_elems * i;
                int len = (i == ntasks - 1) ? (nelems - (per_task_elems * i)) : per_task_elems;

                ElemType partial_res = serial_partial_vdot(v1, start, len);
                sum += partial_res;
            }
        }
    }

    return sum;
}

ElemType ParaVector::norm2(int ntasks) {
    return std::sqrt(parallel_vdot(*this, ntasks));
}

void ParaVector::serial_partial_axpy(const ParaVector& v1, ElemType a, int start, int len) {
    if (std::is_same<ElemType, double>::value) {
        static int incy = 1;
        daxpy(&len, &a, v1.vec.data() + start, &incy, vec.data() + start, &incy);
        return;
    }
    exit(-1);
}

void ParaVector::axpy(const ParaVector& v1, ElemType a, int ntasks) {
    if (v1.nelems != nelems) {
        printf("Got vdot for vectors of different lengths!\n");
        exit(-1);
    }

    int per_task_elems = floor(1.0 * nelems / ntasks);

    ElemType sum = 0;

    #pragma omp taskgroup
    {
        for (int i = 0; i < ntasks; i++) {
            #pragma omp task shared(vec, v1, a, per_task_elems, nelems)
            {
                int start = per_task_elems * i;
                int len = (i == ntasks - 1) ? (nelems - (per_task_elems * i)) : per_task_elems;

                serial_partial_axpy(v1, a, start, len);
            }
        }
    }

    return;
}


void ParaVector::serial_partial_scale(ElemType a, int start, int len) {
    if (len == 0) {
        printf("Len == 0!\n");
        return;
    }

    if (std::is_same<ElemType, double>::value) {
        int incy = 1;
        dscal(&len, &a, vec.data() + start, &incy);
        return;
    }
    exit(-1);
}

void ParaVector::scale(ElemType a, int ntasks) {
    int per_task_elems = floor(1.0 * nelems / ntasks);

    ElemType sum = 0;

    #pragma omp taskgroup
    {
        for (int i = 0; i < ntasks; i++) {
            int start = per_task_elems * i;
            int len = (i == ntasks - 1) ? (nelems - (per_task_elems * i)) : per_task_elems;

            #pragma omp task shared(vec, a)
            serial_partial_scale(a, start, len);
        }
    }

    return;
}

struct CsrMatrix {
    std::vector<ElemType> values;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    int nnz;
    int n;

    CsrMatrix(int _nnz, int _n) : values(_nnz), row_ptr(_n + 1), col_idx(_nnz), nnz(_nnz), n(_n) {}; 
};

std::vector<CsrMatrix> splitMatrices(CsrMatrix &mat, int ntasks) {
    int per_task_lines = floor(1.0 * mat.n / ntasks);

    std::vector<CsrMatrix> vec;

    for (int i = 0; i < ntasks; i++) {
        int start = per_task_lines * i;
        int len = (i == ntasks - 1) ? (mat.n - (per_task_lines * i)) : per_task_lines;

        int n = len;
        int nnz = mat.row_ptr[start + len] - mat.row_ptr[start];
        CsrMatrix m(nnz, n);
        for (int i = 0; i < nnz; i++) {
            m.values[i] = mat.values[i + mat.row_ptr[start]];
            m.col_idx[i] = mat.col_idx[i + mat.row_ptr[start]]; // for mkl
        }
        for (int i = 1; i <= n; i++) {
            m.row_ptr[i] = mat.row_ptr[start + i] - mat.row_ptr[start];
        }

        vec.push_back(m);
    }

    return vec;
}

void printCsrMat(CsrMatrix& m) {
    printf("n: %d, nnz: %d, row_ptr:\n", m.n, m.nnz);
    for (int i = 0; i <= m.n; i++) {
        printf("%d, ", m.row_ptr[i]);
    }
    printf("\ncol_idx:\n");
    for (int i = 0; i < m.nnz; i++) {
        printf("%d, ", m.col_idx[i]);
    }
    printf("\nvalues:\n");
    for (int i = 0; i < m.nnz; i++) {
        printf("%lf, ", m.values[i]);
    }
    printf("\n");
}

ParaVector parallel_spmv(std::vector<CsrMatrix>& mats, ParaVector &v, int nrows) {
    int ntasks = mats.size();

    int nelems = v.nelems;

    int per_task_elems = floor(1.0 * nelems / ntasks);

    ParaVector res(nrows);

    // omp master 起并行区域开销
    // 手动计时单个任务时间，执行个数等信息
    // 用clang llvm openmp

    #pragma omp taskgroup
    {
        for (int i = 0; i < ntasks; i++) {
            #pragma omp task shared(mats, v, res)
            {
                int start = per_task_elems * i;
                int len = (i == ntasks - 1) ? (nelems - (per_task_elems * i)) : per_task_elems;

                // Create a handle for the sparse matrix
                sparse_matrix_t A;
                struct matrix_descr descrA;
                descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
                mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, mats[i].n, v.nelems, mats[i].row_ptr.data(), mats[i].row_ptr.data() + 1, mats[i].col_idx.data(), mats[i].values.data());

                // Perform matrix-vector multiplication
                mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descrA, v.vec.data(), 0.0, res.vec.data() + start);

                // Deallocate the matrix handle
                mkl_sparse_destroy(A);
            }
        }
    }

    return res;
}

struct GMRESData {
    const int max_k = 30;
    const int hh    = (max_k + 2) * (max_k + 1);
    const int hes   = (max_k + 1) * (max_k + 1);
    const int rs    = (max_k + 2);
    const int cc    = (max_k + 1);

    int it = 0;
    int ksp_its = 0;
    int full_cycle = 0;
    ElemType ksp_rnorm = 0;
    ElemType breakdowntol = 0.1;
    int max_it = 300;
    int ksp_reason = 0;

    std::vector<ParaVector> vecs;
    ParaVector vec_temp;
    ParaVector vec_sol;
    std::vector<ElemType> hh_origin;
    std::vector<ElemType> hes_origin;
    std::vector<ElemType> rs_origin;
    std::vector<ElemType> cc_origin;
    std::vector<ElemType> ss_origin;

    GMRESData(int n, int _max_it) : vecs(max_k + 1, ParaVector(n)), vec_temp(n), vec_sol(n), hh_origin(hh), hes_origin(hes), rs_origin(rs), cc_origin(cc), ss_origin(cc), max_it(_max_it) {}
};

#define GMRES_CONJ(x) (x)

void GMRESUpdateHessenberg(GMRESData &gmres, int it, bool hapend, ElemType *res)
{
    ElemType *hh, *cc, *ss, *grs, tt;
    int j;

    hh = gmres.hh_origin.data() + it * (gmres.max_k + 2);
    cc = gmres.cc_origin.data();
    ss = gmres.ss_origin.data();
    grs = gmres.rs_origin.data();

    /* Apply all the previously computed plane rotations to the new column
        of the Hessenberg matrix */
    for (j = 1; j <= it; j++) {
        tt  = *hh;
        *hh = GMRES_CONJ(*cc) * tt + *ss * *(hh + 1);
        hh++;
        *hh = *cc++ * *hh - (*ss++ * tt);
    }

    /*
        compute the new plane rotation, and apply it to:
        1) the right-hand-side of the Hessenberg system
        2) the new column of the Hessenberg matrix
        thus obtaining the updated value of the residual
    */
    if (!hapend) {
        printf("hh: %lf, hh+1: %lf\n", *(hh + 0), *(hh + 1));
        tt = std::sqrt(GMRES_CONJ(*hh) * *hh + GMRES_CONJ(*(hh + 1)) * *(hh + 1));
        if (tt == 0.0) {  
            // PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "tt == 0.0");
            gmres.ksp_reason = 5;
            return;
        }
        *cc          = *hh / tt;
        *ss          = *(hh + 1) / tt;
        grs[it + 1] = -(*ss * grs[it]);
        grs[it]     = GMRES_CONJ(*cc) * grs[it];
        *hh          = GMRES_CONJ(*cc) * *hh + *ss * *(hh + 1);
        *res         = std::abs(grs[it + 1]);
    } else {
        /* happy breakdown */
        *res = 0.0;
    }
}

void KSPGMRESBuildSoln(GMRESData &gmres, ElemType *nrs, ParaVector &vs, ParaVector &vdest, int it, int ntasks)
{
    ElemType tt;
    int    ii, k, j;

    /* Solve for solution vector that minimizes the residual */

    /* If it is < 0, no gmres steps have been performed */
    if (it < 0) {
        vdest = vs;
        return;
    }
    if (gmres.hh_origin[it * (gmres.max_k + 2) + (it)] != 0.0) {
        nrs[it] = gmres.rs_origin[it] / gmres.hh_origin[it * (gmres.max_k + 2) + (it)];
    } else {
        gmres.ksp_reason = 3;

        printf("Likely your matrix or preconditioner is singular. HH(it,it) is identically zero; it = %d\n", it);
        return;
    }
    for (ii = 1; ii <= it; ii++) {
        k  = it - ii;
        tt = gmres.rs_origin[k];
        for (j = k + 1; j <= it; j++) {
            tt = tt - gmres.hh_origin[j * (gmres.max_k + 2) + (k)] * nrs[j];
        }
        if (gmres.hh_origin[k * (gmres.max_k + 2) + (k)] == 0.0) {
            gmres.ksp_reason = 4;
            printf("Likely your matrix or preconditioner is singular. HH(k,k) is identically zero; k = %d\n", k);
            return;
        }
        nrs[k] = tt / gmres.hh_origin[k * (gmres.max_k + 2) + (k)];
    }

    /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
    gmres.vec_temp.set_val(0.0);
    
    // PetscCall(VecMAXPY(VEC_TEMP, it + 1, nrs, &VEC_VV(0)));
    for (int i = 0; i < it + 1; i++) {
        gmres.vec_temp.axpy(gmres.vecs[i], nrs[i], ntasks);
    }

    // PetscCall(KSPUnwindPreconditioner(ksp, VEC_TEMP, VEC_TEMP_MATOP));
    // /* add solution to previous solution */
    // if (vdest != vs) PetscCall(VecCopy(vs, vdest));
    vdest = vs;
    vdest.axpy(gmres.vec_temp, 1.0, ntasks);
}

void GMRESCycle(CsrMatrix &mat, vector<CsrMatrix> &mats, ParaVector &rhs, GMRESData &gmres, int ntasks, int *itcount) {
    // std::vector<std::vector<int>> dep_array(gmres.max_k + 1, std::vector<int>(ntasks, 0));
    int dep_array[gmres.max_k + 1][ntasks];
    int nrows = mat.n;

    // setup data ptr
    ElemType *hh_base = gmres.hh_origin.data();
    ElemType *hes_base = gmres.hes_origin.data();

    ElemType hapbnd, tt;
    int it = 0;
    bool hapend = false;

    ElemType res = rhs.norm2(ntasks);

    if ((gmres.ksp_rnorm > 0.0) && (std::fabs(res - gmres.ksp_rnorm) > gmres.breakdowntol * gmres.ksp_rnorm) && (!ignore_ksp_stop)) {
        printf("Residual norm computed by GMRES recursion formula xxx is far from the computed residual norm xxx at restart, residual norm at start of cycle xx");
        gmres.ksp_reason = 1;
        return;
    }

    // check if converged
    gmres.rs_origin[0] = res;
    gmres.ksp_rnorm = res;
    gmres.it = (it - 1);
    if (!res) {
        gmres.ksp_reason = 2;
        printf("Converged due to zero residual norm on entry, res_norm = %f\n", res);
        return;
    }
    // check ksp->converged

    // get data for partition
    int per_task_elems = floor(1.0 * nrows / ntasks);

    while (it < gmres.max_k) {
        // apply PC
        // gmres.vecs[it + 1] = parallel_spmv(mats, gmres.vecs[it], nrows);
        // #pragma omp taskgroup
        {
            for (int i = 0; i < ntasks; i++) {
                #pragma omp task affinity(dep_array[it + 1][i]) shared(mats, gmres, per_task_elems, ntasks, it) depend(in: dep_array[it][i]) depend(out: dep_array[it + 1][i]) 
                {
                    int start = per_task_elems * i;
                    // Create a handle for the sparse matrix
                    sparse_matrix_t A;
                    struct matrix_descr descrA;
                    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
                    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, mats[i].n, gmres.vecs[it].nelems, mats[i].row_ptr.data(), mats[i].row_ptr.data() + 1, mats[i].col_idx.data(), mats[i].values.data());

                    // Perform matrix-vector multiplication
                    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descrA, gmres.vecs[it].vec.data(), 0.0, gmres.vecs[it + 1].vec.data() + start);

                    // Deallocate the matrix handle
                    mkl_sparse_destroy(A);
                }
            }
        }
        
        // orthog process
        // GMRESModifiedGramSchmidtOrthogonalization(gmres, it, ntasks);
        int     orthog_j;
        ElemType *orthog_hh, *orthog_hes;

        /* update Hessenberg matrix and do Gram-Schmidt */
        orthog_hh  = gmres.hh_origin.data() + (it) * (gmres.max_k + 2);
        orthog_hes = gmres.hes_origin.data() + (it) * (gmres.max_k + 1);
        for (int j = 0; j <= it; j++) {
            /* (vv(it+1), vv(j)) */
            
            // PetscCall(VecDot(VEC_VV(it + 1), VEC_VV(j), hh));
            // *orthog_hh = gmres.vecs[it+1].parallel_vdot(gmres.vecs[j], ntasks);
            ElemType sum = 0;

            #pragma omp taskgroup task_reduction(+:sum)
            {
                for (int i = 0; i < ntasks; i++) {
                    #pragma omp task in_reduction(+:sum) affinity(dep_array[it + 1][i]) shared(gmres, per_task_elems, ntasks, it) depend(in: dep_array[it + 1][i], dep_array[j][i])
                    {
                        int start = per_task_elems * i;
                        int len = (i == ntasks - 1) ? (gmres.vecs[it+1].nelems - (per_task_elems * i)) : per_task_elems;
                        ElemType partial_res = gmres.vecs[it+1].serial_partial_vdot(gmres.vecs[j], start, len);
                        sum += partial_res;
                    }
                }
            }


            *orthog_hh = sum;

            *orthog_hes++ = *orthog_hh;
            /* vv(it+1) <- vv(it+1) - hh[it+1][j] vv(j) */
            // PetscCall(VecAXPY(VEC_VV(it + 1), -(*hh++), VEC_VV(j)));
            // gmres.vecs[it+1].axpy(gmres.vecs[j], -(*orthog_hh++), ntasks);
            // #pragma omp taskgroup
            {
                ElemType a = -(*orthog_hh++);
                for (int i = 0; i < ntasks; i++) {
                    
                    #pragma omp task affinity(dep_array[it + 1][i]) shared(gmres, a, per_task_elems, ntasks, it) depend(in: dep_array[j][i]) depend(inout: dep_array[it + 1][i]) 
                    {
                        int start = per_task_elems * i;
                        int len = (i == ntasks - 1) ? (gmres.vecs[it+1].nelems - (per_task_elems * i)) : per_task_elems;
                        gmres.vecs[it+1].serial_partial_axpy(gmres.vecs[j], a, start, len);
                    }
                }
            }
        }


        if (gmres.ksp_reason && (!ignore_ksp_stop)) break;

        #pragma omp taskwait

        // vector normalize
        {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            tt = gmres.vecs[it + 1].norm2(ntasks);
            printf("tt: %lf\n", tt);

            // ==============================================
            // -> scale vector
            // gmres.vecs[it + 1].scale(1.0 / tt, ntasks);
            // #pragma omp taskgroup
        
            for (int i = 0; i < ntasks; i++) {
                int start = per_task_elems * i;
                int len = (i == ntasks - 1) ? (nrows - (per_task_elems * i)) : per_task_elems;

                #pragma omp task shared(gmres, tt) depend(inout: dep_array[it + 1][i]) 
                gmres.vecs[it + 1].serial_partial_scale(1.0 / tt, start, len);
            }
        }

        /* save the magnitude */
        hh_base[it * (gmres.max_k + 2) + (it + 1)]  = tt;
        hes_base[it * (gmres.max_k + 1) + (it + 1)] = tt;

        /* check for the happy breakdown */
        hapbnd = std::fabs(tt / gmres.rs_origin[it]);
        // if (hapbnd > gmres->haptol) hapbnd = gmres.haptol;
        if (tt < hapbnd && (!ignore_ksp_stop)) {
            printf("Detected happy breakdown, current hapbnd = %14.12e tt = %14.12e\n", (double)hapbnd, (double)tt);
            hapend = true;
        }

        #pragma omp taskwait

        // serial process
        GMRESUpdateHessenberg(gmres, it, hapend, &res);

        it++;
        gmres.it = (it - 1); /* For converged */
        gmres.ksp_its++;
        gmres.ksp_rnorm = res;

        printf("finish all process, its: %d\n", gmres.ksp_its);

        if (gmres.ksp_reason && (!ignore_ksp_stop)) break;
        // todo:
        // PetscCall((*ksp->converged)(ksp, ksp->its, res, &ksp->reason, ksp->cnvP));

        /* omitted: Catch error in happy breakdown and signal convergence and break from loop */
    }

    if (itcount) *itcount = it;
    // KSPGMRESBuildSoln(GRS(0), ksp->vec_sol, ksp->vec_sol, ksp, it - 1);
    KSPGMRESBuildSoln(gmres, gmres.rs_origin.data(), gmres.vec_sol, gmres.vec_sol, it - 1, ntasks);
    return;
}

void solve_gmres(CsrMatrix &mat, ParaVector &rhs, GMRESData &gmres, int ntasks) {
    auto mats = splitMatrices(mat, ntasks);

    int its, itcount;
    int N =  gmres.max_k + 1;
    gmres.ksp_its = 0;

    itcount = 0;
    gmres.full_cycle = 0;
    gmres.ksp_rnorm = -1;

    while (true) {
        // KSPInitialResidual
        gmres.vecs[0] = rhs;
        // KSPGMRESCycle
        GMRESCycle(mat, mats, rhs, gmres, ntasks, &its);

        if (its == gmres.max_k) {
            gmres.full_cycle++;
        }
        itcount += its;
        printf("Its %d\n", itcount);
        if (itcount >= gmres.max_it) {
            // reach max its, record and quit
            break;
        }
    }
}

int main()
{
    const char* filename = "/workspace/matrices/rajat21/rajat21.mtx";
    
    CSRMatrixInfo* matrix = readCSRMatrix(filename);

    CsrMatrix m(matrix->nnz, matrix->rows);
    for (int i = 0; i < m.n + 1; i++) {
        m.row_ptr[i] = matrix->row_ptr[i];
    }
    for (int i = 0; i < m.nnz; i++) {
        m.col_idx[i] = matrix->col_idx[i];
        m.values[i] = matrix->values[i];
    }

    ParaVector vec(m.n);
    for (int i = 0; i < vec.nelems; i++) {
        vec.vec[i] = 0.5 + (i / 338610.0);
    }

    GMRESData gmres(m.n, 1200);

    #pragma omp parallel
    {
        #pragma omp master
        {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            solve_gmres(m, vec, gmres, 32);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            // 计算时间间隔
            std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            std::cout << "Elapsed time: " << duration.count() << " seconds\n";
        }
    }

    return 0;
}
