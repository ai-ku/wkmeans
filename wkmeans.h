#ifndef __WKMEANS_H__
#define __WKMEANS_H__


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

typedef int bool;
#define true 1
#define false 0
#define SQRT2 1.4142135623731
#define BUF (1<<16)

PREC kmeans(PREC *CX, const PREC *X, PREC *W, unsigned int *assignment, unsigned int dim, unsigned int npts, unsigned int nclus, unsigned int maxiter, unsigned int restarts);
void kmeans_error(char *msg);
int comp_randperm (const void * a, const void * b);
void randperm(unsigned int *order, unsigned int npoints);
PREC compute_distance(const PREC *vec1, const PREC *vec2, const unsigned int dim);
PREC compute_rms(const PREC *CX, const PREC *X, const PREC *W, const unsigned int *c, unsigned int dim, unsigned int npts, unsigned int ncls);
PREC compute_rms1(const PREC *CX, const PREC *X, const PREC *W, const unsigned int *c, unsigned int dim, unsigned int npts);
PREC compute_rms2(const PREC *CX, unsigned int dim, unsigned int ncls);
void remove_point_from_cluster(unsigned int cluster_ind, PREC *CX, const PREC *px, PREC pw, unsigned int *nr_points, PREC *CW, unsigned int dim);
void add_point_to_cluster(unsigned int cluster_ind, PREC *CX, const PREC *px, PREC pw, unsigned int *nr_points, PREC *CW, unsigned int dim);
bool remove_identical_clusters(PREC *CX, BOUND_PREC *cluster_distance, const PREC *X, const PREC *W, unsigned int *cluster_count, PREC *CW, unsigned int *c, unsigned int dim, unsigned int nclus, unsigned int npts);
void compute_cluster_distances(BOUND_PREC *dist, BOUND_PREC *s, const PREC *CX, unsigned int dim, unsigned int nclus, const bool *cluster_changed);
unsigned int init_point_to_cluster(unsigned int point_ind, const PREC *px, const PREC *CX, unsigned int dim, unsigned int nclus, PREC *mindist, BOUND_PREC *low_b, const BOUND_PREC *cl_dist);
unsigned int assign_point_to_cluster_ordinary(const PREC *px, const PREC *CX, unsigned int dim, unsigned int nclus);
unsigned int assign_point_to_cluster(unsigned int point_ind, const PREC *px, const PREC *CX, unsigned int dim, unsigned int nclus, unsigned int old_assignment, PREC *mindist, BOUND_PREC *s, BOUND_PREC *cl_dist, BOUND_PREC *low_b);
PREC kmeans_run(PREC *CX, const PREC *X, const PREC *W, unsigned int *c, unsigned int dim, unsigned int npts, unsigned int nclus, unsigned int maxiter);
void furthest_first (double *CX, const double *X, unsigned int dim, unsigned int npts, unsigned int nclus);
void furthest_first_sample (double *CX, const double *X, double *W, unsigned int dim, unsigned int npts, unsigned int nclus);
void rand_ff (double *CX, const double *X, unsigned int dim, unsigned int npts, unsigned int nclus);
void kpp (double *CX, const double *X, double *W, unsigned int dim, unsigned int npts, unsigned int nclus);
void random_init (double *CX, const double *X, unsigned int dim, unsigned int npts, unsigned int nclus);

#endif


