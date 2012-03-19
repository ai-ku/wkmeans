#ifndef __MPI_KMEANS_MEX_H__
#define __MPI_KMEANS_MEX_H__


#ifndef KMEANS_VERBOSE 
#define KMEANS_VERBOSE 0
#endif

/* Double precision is default*/
#ifndef INPUT_TYPE
#define INPUT_TYPE 0
#endif

#if INPUT_TYPE==0
#define PREC double
#define PREC_MAX DBL_MAX
#elif INPUT_TYPE==1
#define PREC float
#define PREC_MAX FLT_MAX
#endif


#ifndef BOUND_PREC
#define BOUND_PREC float
#endif

#ifndef BOUND_PREC_MAX
#define BOUND_PREC_MAX FLT_MAX
#endif

#define BOUND_EPS 1e-6

extern "C"{
PREC kmeans(PREC *CXp,const PREC *X,unsigned int *c,unsigned int dim,unsigned int npts,unsigned int nclus,unsigned int maxiter, unsigned int nr_restarts);
}
PREC compute_distance(const PREC *vec1, const PREC *vec2, const unsigned int dim);
unsigned int assign_point_to_cluster_ordinary(const PREC *px, const PREC *CX, unsigned int dim,unsigned int nclus);
void randperm(unsigned int *order, unsigned int npoints);

#endif


