/** wkmeans: k-means algorithm with (optional) instance weights.
 *  Based on mpi_kmeans-1.5 by Peter Gehler.
 *  Based on C. Elkan. Using the triangle inequality to accelerate kMeans. ICML 2003.
 *  Initialization based on Arthur, D. and Vassilvitskii,
 *  S. (2007). K-means++: the advantages of careful seeding.
 *  Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
 *  algorithms. pp. 1027-1035. 
 *  Last modified by Deniz Yuret and Enis Sert, 25-Mar-2012.
 */

const char *rcsid = "$Id$";
const char *usage = "wkmeans [options] < input > output\n"
  "-k number of clusters (default 2)\n"
  "-r number of restarts (default 0)\n"
  "-s random seed\n"
  "-l input file contains labels\n"
  "-w input file contains instance weights\n"
  "-v verbose output\n";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <memory.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include "wkmeans.h"

#if KMEANS_VERBOSE>1
unsigned int saved_two=0, saved_three_one=0, saved_three_two=0, saved_three_three=0, saved_three_b=0;
#endif

int VERBOSE = 0;


int main(int argc, char **argv) {
  int nof_clusters = 2;
  int nof_restarts = 0;
  int maxiter = 0;
  int weights = 0;
  int labels = 0;
  unsigned int seed = 0;
  int opt;
  while((opt = getopt(argc, argv, "k:r:i:s:lwvh")) != -1) {
    switch(opt) {
    case 'k': nof_clusters = atoi(optarg); break;
    case 'r': nof_restarts = atoi(optarg); break;
    case 's': seed = atoi(optarg); break;
    case 'l': labels = 1; break;
    case 'w': weights = 1; break;
    case 'v': VERBOSE = 1; break;
    default: fputs(usage, stderr); exit(0);
    }
  }
  if (seed) srand(seed);

  PREC *X = NULL;
  PREC *W = NULL;
  char **L = NULL;
  PREC *CX = NULL;
  unsigned int *assignment = NULL;
  unsigned int dims = 0;
  unsigned int nof_points = 0;

  int nx = BUF;
  int ix = 0;
  X = malloc(nx * sizeof(PREC));

  int nw = BUF;
  int iw = 0;
  W = malloc(nw * sizeof(PREC));

  int nl = BUF;
  int il = 0;
  L = malloc(nl * sizeof(char*));

  char buf[BUF];
  int row = 0;
  while(fgets(buf, BUF, stdin) != NULL) {
    int col = 0;
    for (char *ptr = strtok(buf, " \t\n\r\f\v"); ptr != NULL; ptr = strtok(NULL, " \t\n\r\f\v")) {
      PREC x = atof(ptr);
      if (labels && il <= row) {
	L[il++] = strdup(ptr);
      } else if (weights && iw <= row) {
	W[iw++] = x;
      } else {
	X[ix++] = x;
	if (!weights && iw <= row) W[iw++] = 1;
      }
      if (ix == nx) {
	nx *= SQRT2;
	X = realloc(X, nx * sizeof(PREC));
      }
      col++;
    }
    if (row == 0) {
      dims = ix;
    }
    if (iw == nw) {
      nw *= SQRT2;
      W = realloc(W, nw * sizeof(PREC));
    }
    if (il == nl) {
      nl *= SQRT2;
      L = realloc(L, nl * sizeof(char*));
    }
    row++;
  }
  nof_points = iw;
  
  if (VERBOSE) {
    fprintf(stderr, "Read %d points in %d dimensions%s%s.\n", 
	    nof_points, dims, 
	    (labels ? " with labels" : ""),
	    (weights ? " with weights" : ""));
  }

  assignment = calloc(nof_points, sizeof(unsigned int));
  CX = calloc(dims * nof_clusters, sizeof(PREC));
  PREC rms = kmeans(CX, X, W, assignment, dims, nof_points, nof_clusters, maxiter, nof_restarts);

  for (int i = 0; i < nof_points; i++) {
    if (labels) {
      printf("%s\t", L[i]);
      free(L[i]);
    }
    printf("%d\n", assignment[i]);
  }

  fprintf(stderr, "%f\n", rms);

  free(X);
  free(W);
  free(L);
  free(assignment);
  free(CX);
}


PREC kmeans(PREC *CX, const PREC *X, PREC *W, unsigned int *assignment, unsigned int dim, unsigned int npts, unsigned int nclus, unsigned int maxiter, unsigned int restarts)
{

  if (npts < nclus)
    {
      CX = (PREC*)calloc(nclus*dim, sizeof(PREC));
      memcpy(CX, X, dim*nclus*sizeof(PREC));
      PREC rms = 0.0;
      return(rms);
    }
  else if (npts == nclus)
    {
      memcpy(CX, X, dim*nclus*sizeof(PREC));
      PREC rms = 0.0;
      return(rms);
    }
  else if (nclus == 0)
    {
      printf("Error: Number of clusters is 0\n");
      exit(-1);
    }

  if (W == NULL) {		// DY: If no weights specified initialize all weigths to 1.
    W = (PREC*) malloc(npts*sizeof(PREC));
    for (int i = 0; i < npts; i++) 
      W[i] = 1.0;
  }

  kpp(CX, X, W, dim, npts, nclus);
  PREC rms = kmeans_run(CX, X, W, assignment, dim, npts, nclus, maxiter);
  if (VERBOSE) fprintf(stderr, "Iteration %d/%d rms: %f\n", 1, restarts, rms);

  unsigned int res = restarts - 1;
  if (res>0)
    {
      PREC minrms = rms;
      unsigned int *order = (unsigned int*)malloc(npts*sizeof(unsigned int));
      PREC *bestCX = (PREC*) malloc(dim*nclus*sizeof(PREC));
      unsigned int *bestassignment = (unsigned int*)malloc(npts*sizeof(unsigned int));

      memcpy(bestCX, CX, dim*nclus*sizeof(PREC));
      memcpy(bestassignment, assignment, npts*sizeof(unsigned int));

      while (res>0)
	{

	  kpp(CX, X, W, dim, npts, nclus);
	  rms = kmeans_run(CX, X, W, assignment, dim, npts, nclus, maxiter);
	  if (VERBOSE) fprintf(stderr, "Iteration %d/%d rms: %f\n", 1+restarts-res, restarts, rms);
	  if (rms<minrms)
	    {
	      if (VERBOSE) fprintf(stderr, "Updating best clustering rms = %g\n", rms);
	      minrms = rms;
	      memcpy(bestCX, CX, dim*nclus*sizeof(PREC));
	      memcpy(bestassignment, assignment, npts*sizeof(unsigned int));
	    }
	  res--;

	}
      memcpy(CX, bestCX, dim*nclus*sizeof(PREC));
      memcpy(assignment, bestassignment, npts*sizeof(unsigned int));
      rms = minrms;
      free(bestassignment);
      free(bestCX);
      free(order);
    }
  assert(CX != NULL);

  return(rms);

}


void kmeans_error(char *msg)
{
  printf("%s", msg);
  exit(-1);
}

int comp_randperm (const void * a, const void * b)
{
  return ((int)( *(double*)a - *(double*)b ));
}


void randperm(unsigned int *order, unsigned int npoints)
{
  double *r = (double*)malloc(2*npoints*sizeof(double));
  for (unsigned int i=0; i<2*npoints; i++, i++)
    {
      r[i] = rand();
      r[i+1] = i/2;
    }
  qsort (r, npoints, 2*sizeof(double), comp_randperm);

  for (unsigned int i=1; i<2*npoints; i++, i++)
    order[i/2] = (unsigned int)r[i];

  free(r);
}

PREC compute_distance(const PREC *vec1, const PREC *vec2, const unsigned int dim)
{
  PREC d = 0.0;
  for ( unsigned int k=0 ; k<dim ; k++ )
    {
      PREC df = (vec1[k]-vec2[k]);
      d += df*df;
    }
  assert(d>=0.0);
  d = sqrt(d);

  return d;
}

PREC compute_rms(const PREC *CX, const PREC *X, const PREC *W, const unsigned int *c, unsigned int dim, unsigned int npts, unsigned int ncls) {
  PREC rms1 = compute_rms1(CX, X, W, c, dim, npts);
  PREC rms2 = compute_rms2(CX, dim, ncls);
  return (rms1 / rms2);
}

/* This computes the within cluster root-mean-squared distance */

PREC compute_rms1(const PREC *CX, const PREC *X, const PREC *W, const unsigned int *c, unsigned int dim, unsigned int npts)
{
  PREC sum = 0.0;
  PREC rms = 0.0;
  const PREC *px = X;
  for ( unsigned int i=0 ; i<npts ; i++, px+=dim)
    {
      const PREC *pcx = CX+c[i]*dim;
      PREC d = compute_distance(px, pcx, dim);
      rms += W[i]*d*d;	// DY: we just took sqrt, this is inefficient, need sqdist fn
      sum += W[i];
    }
  rms /= sum;
  rms = sqrt(rms);
  assert(rms>=0.0);
  return(rms);
}

/* This computes the between cluster root-mean-squared distance */

PREC compute_rms2(const PREC *CX, unsigned int dim, unsigned int ncls)
{
  int cnt = 0;
  PREC rms = 0.0;
  for (int i = ncls - 1; i > 0; i--) {
    for (int j = i - 1; j >= 0; j--) {
      const PREC *pc1 = CX + i * dim;
      const PREC *pc2 = CX + j * dim;
      PREC d = compute_distance(pc1, pc2, dim);
      rms += d * d;
      cnt++;
    }
  }
  rms /= cnt;
  rms = sqrt(rms);
  return (rms);
}

void remove_point_from_cluster(unsigned int cluster_ind, PREC *CX, const PREC *px, PREC pw, unsigned int *nr_points, PREC *CW, unsigned int dim)
{
  PREC *pcx = CX + cluster_ind*dim; // DY: centroid coordinates

  /* empty cluster after or before removal */
  if (nr_points[cluster_ind]<2) // DY: why not == 1?
    {
      for ( unsigned int k=0 ; k<dim ; k++ )
	pcx[k] = 0.0; // DY: why zero out the coordinates?  to compute new average?
      nr_points[cluster_ind]=0;
      CW[cluster_ind] = 0;
    }
  else
    {
      /* pgehler: remove PREC here */
      PREC cw_old, cw_new; 
      cw_old = CW[cluster_ind]; // DY: this could be sum of weights
      (nr_points[cluster_ind])--;	       // DY: subtract point weight
      CW[cluster_ind] -= pw;
      cw_new = CW[cluster_ind]; // DY: this should be the new sum of weights

      for ( unsigned int k=0 ; k<dim ; k++ )
	pcx[k] = (cw_old*pcx[k] - pw*px[k])/cw_new; // DY: subtract the weight of point times its coord.
    }
}

void add_point_to_cluster(unsigned int cluster_ind, PREC *CX, const PREC *px, PREC pw, unsigned int *nr_points, PREC *CW, unsigned int dim)
{

  PREC *pcx = CX + cluster_ind*dim;

  /* first point in cluster */
  if (nr_points[cluster_ind]==0)
    {		
      (nr_points[cluster_ind])++; // DY: this should be incremented by weight of point
      CW[cluster_ind] = pw;
      for ( unsigned int k=0 ; k<dim ; k++ )
	pcx[k] = px[k]; // DY: see, no need to zero out an empty cluster.
    }
  else
    {
      /* remove PREC here */
      PREC cw_old = CW[cluster_ind]; // DY: add weights same as the remove point code
      (nr_points[cluster_ind])++;
      CW[cluster_ind] += pw;
      PREC cw_new = CW[cluster_ind];
      for ( unsigned int k=0 ; k<dim ; k++ )
	pcx[k] = (cw_old*pcx[k]+pw*px[k])/cw_new;
    }
}


bool remove_identical_clusters(PREC *CX, BOUND_PREC *cluster_distance, const PREC *X, const PREC *W, unsigned int *cluster_count, PREC *CW, unsigned int *c, unsigned int dim, unsigned int nclus, unsigned int npts)
{
  bool stat = false;
  for ( unsigned int i=0 ; i<(nclus-1) ; i++ )
    {
      for ( unsigned int j=i+1 ; j<nclus ; j++ )
	{
	  if (cluster_distance[i*nclus+j] <= BOUND_EPS)
	    {
#if KMEANS_VERBOSE>1
	      printf("found identical cluster : %d\n", j);
#endif
	      stat = true;
	      /* assign the points from j to i */
	      const PREC *px = X;
	      for ( unsigned int n=0 ; n<npts ; n++, px+=dim )
		{
		  if (c[n] != j) continue; // DY: c[n] is uninitialized at this point!!!
		  remove_point_from_cluster(j, CX, px, W[n], cluster_count, CW, dim); // DY: say j instead of c[n]
		  c[n] = i;
		  add_point_to_cluster(i, CX, px, W[n], cluster_count, CW, dim);
		}
	    }
	}
    }
  return(stat);		// this just makes n-1 of the identical clusters empty.
}

void compute_cluster_distances(BOUND_PREC *dist, BOUND_PREC *s, const PREC *CX, unsigned int dim, unsigned int nclus, const bool *cluster_changed)
{
  for ( unsigned int j=0 ; j<nclus ; j++ )
    s[j] = BOUND_PREC_MAX;

  const PREC *pcx = CX;
  for ( unsigned int i=0 ; i<nclus-1 ; i++, pcx+=dim)
    {
      const PREC *pcxp = CX + (i+1)*dim;
      unsigned int cnt=i*nclus+i+1;
      for ( unsigned int j=i+1 ; j<nclus; j++, cnt++, pcxp+=dim )
	{
	  if (cluster_changed[i] || cluster_changed[j]) // DY: Update dist and s if clusters changed?
	    {
	      dist[cnt] = (BOUND_PREC)(0.5 * compute_distance(pcx, pcxp, dim));
	      dist[j*nclus+i] = dist[cnt];

	      if (dist[cnt] < s[i])
		s[i] = dist[cnt];

	      if (dist[cnt] < s[j])
		s[j] = dist[cnt];
	    }
	}
    }
}


unsigned int init_point_to_cluster(unsigned int point_ind, const PREC *px, const PREC *CX, unsigned int dim, unsigned int nclus, PREC *mindist, BOUND_PREC *low_b, const BOUND_PREC *cl_dist)
{
  bool use_low_b = true;

  if (low_b==NULL) use_low_b = false;
  unsigned int bias = point_ind*nclus;
	
  const PREC *pcx = CX;
  PREC mind = compute_distance(px, pcx, dim);
  if (use_low_b) low_b[bias] = (BOUND_PREC)mind;
  unsigned int assignment = 0;
  pcx+=dim;
  for ( unsigned int j=1 ; j<nclus ; j++, pcx+=dim )
    {
      if (mind + BOUND_EPS <= cl_dist[assignment*nclus+j])
	continue;

      PREC d = compute_distance(px, pcx, dim);
      if(use_low_b) low_b[j+bias] = (BOUND_PREC)d;

      if (d<mind)
	{
	  mind = d;
	  assignment = j;
	}
    }
  mindist[point_ind] = mind;
  return(assignment);
}

unsigned int assign_point_to_cluster_ordinary(const PREC *px, const PREC *CX, unsigned int dim, unsigned int nclus)
{
  unsigned int assignment = nclus;
  PREC mind = PREC_MAX;
  const PREC *pcx = CX;
  for ( unsigned int j=0 ; j<nclus ; j++, pcx+=dim )
    {
      PREC d = compute_distance(px, pcx, dim);
      if (d<mind)
	{
	  mind = d;
	  assignment = j;
	}
    }
  assert(assignment < nclus);
  return(assignment);
}

unsigned int assign_point_to_cluster(unsigned int point_ind, const PREC *px, const PREC *CX, unsigned int dim, unsigned int nclus, unsigned int old_assignment, PREC *mindist, BOUND_PREC *s, BOUND_PREC *cl_dist, BOUND_PREC *low_b)
{
  bool up_to_date = false, use_low_b=true;;

  unsigned int bias = point_ind*nclus;
  if (low_b==NULL)use_low_b=false;

  PREC mind = mindist[point_ind];

  if (mind+BOUND_EPS <= s[old_assignment])
    {
#ifdef KMEANS_VEBOSE
      saved_two++;
#endif
      return(old_assignment);
    }

  unsigned int assignment = old_assignment;
  unsigned int counter = assignment*nclus;
  const PREC *pcx = CX;
  for ( unsigned int j=0 ; j<nclus ; j++, pcx+=dim )
    {
      if (j==old_assignment)
	{
#if KMEANS_VERBOSE>1
	  saved_three_one++;
#endif
	  continue;
	}
		
      if (use_low_b && (mind+BOUND_EPS <= low_b[j+bias]))
	{
#if KMEANS_VERBOSE>1
	  saved_three_two++;
#endif
	  continue;
	}

      if (mind+BOUND_EPS <= cl_dist[counter+j])
	{
#if KMEANS_VERBOSE>1
	  saved_three_three++;
#endif
	  continue;
	}

      PREC d = 0.0;
      if (!up_to_date)
	{
	  d = compute_distance(px, CX+assignment*dim, dim);
	  mind = d;
	  if(use_low_b) low_b[assignment+bias] = (BOUND_PREC)d;
	  up_to_date = true;
	}
		
      if (!use_low_b)
	d = compute_distance(px, pcx, dim);
      else if ((mind > BOUND_EPS+low_b[j+bias]) || (mind > BOUND_EPS+cl_dist[counter+j]))
	{
	  d =compute_distance(px, pcx, dim);
	  low_b[j+bias] = (BOUND_PREC)d;
	}
      else
	{
#if KMEANS_VERBOSE>1
	  saved_three_b++;
#endif
	  continue;
	}

      if (d<mind)
	{
	  mind = d;
	  assignment = j;
	  counter = assignment*nclus;
	  up_to_date = true;
	}
    }
  mindist[point_ind] = mind;

  return(assignment);
}


PREC kmeans_run(PREC *CX, const PREC *X, const PREC *W, unsigned int *c, unsigned int dim, unsigned int npts, unsigned int nclus, unsigned int maxiter)
{
  PREC *tCX = (PREC *)calloc(nclus * dim, sizeof(PREC));
  if (tCX==NULL)	kmeans_error((char*)"Failed to allocate mem for Cluster points");

  /* number of points per cluster */
  unsigned int *CN = (unsigned int *) calloc(nclus, sizeof(unsigned int)); 
  if (CN==NULL)	kmeans_error((char*)"Failed to allocate mem for assignment");
	
  /* total weight of points in cluster */
  PREC *CW = (PREC *) calloc(nclus, sizeof(PREC)); 
  if (CW==NULL)	kmeans_error((char*)"Failed to allocate mem for cluster weights");
	
  /* old assignement of points to cluster */
  unsigned int *old_c = (unsigned int *) malloc(npts* sizeof(unsigned int));
  if (old_c==NULL)	kmeans_error((char*)"Failed to allocate mem for temp assignment");

  /* assign to value which is out of range */
  for ( unsigned int i=0 ; i<npts ; i++)
    old_c[i] = nclus;

#if KMEANS_VERBOSE>0
  printf("compile without setting the KMEANS_VERBOSE flag for no output\n");
#endif

  BOUND_PREC *low_b = (BOUND_PREC *) calloc(npts*nclus, sizeof(BOUND_PREC));
  bool use_low_b = false;
  if (low_b == NULL)
    {
#if KMEANS_VERBOSE>0
      printf("not enough memory for lower bound, will compute without\n");
#endif
      use_low_b = false;
    }
  else
    {
      use_low_b = true;
      assert(low_b);
    }


  BOUND_PREC *cl_dist = (BOUND_PREC *)calloc(nclus*nclus, sizeof(BOUND_PREC));
  if (cl_dist==NULL)	kmeans_error((char*)"Failed to allocate mem for cluster-cluster distance");

  BOUND_PREC *s = (BOUND_PREC *) malloc(nclus*sizeof(BOUND_PREC));
  if (s==NULL)	kmeans_error((char*)"Failed to allocate mem for assignment");

  BOUND_PREC *offset = (BOUND_PREC *) malloc(nclus * sizeof(BOUND_PREC)); /* change in distance of a cluster mean after a iteration */
  if (offset==NULL)	kmeans_error((char*)"Failed to allocate mem for bound points-nearest cluster");

  PREC *mindist = (PREC *)malloc(npts * sizeof(PREC));
  if (mindist==NULL)	kmeans_error((char*)"Failed to allocate mem for bound points-clusters");

  for ( unsigned int i=0;i<npts;i++)
    mindist[i] = PREC_MAX;

  bool *cluster_changed = (bool *) malloc(nclus * sizeof(bool)); /* did the cluster changed? */
  if (cluster_changed==NULL)	kmeans_error((char*)"Failed to allocate mem for variable cluster_changed");
  for ( unsigned int j=0 ; j<nclus ; j++ )
    cluster_changed[j] = true;


  unsigned int iteration = 0;
  unsigned int nchanged = 1;
  while (iteration < maxiter || maxiter == 0)
    {
		
      /* compute cluster-cluster distances */
      compute_cluster_distances(cl_dist, s, CX, dim, nclus, cluster_changed);
		
      /* assign all points from identical clusters to the first occurence of that cluster */
      remove_identical_clusters(CX, cl_dist, X, W, CN, CW, c, dim, nclus, npts);
			
      /* find nearest cluster center */
      if (iteration == 0)
	{
		  
	  const PREC *px = X;
	  for ( unsigned int i=0 ; i<npts ; i++, px+=dim)
	    {
	      c[i] = init_point_to_cluster(i, px, CX, dim, nclus, mindist, low_b, cl_dist);
	      add_point_to_cluster(c[i], tCX, px, W[i], CN, CW, dim);
	    }
	  nchanged = npts;
	}
      else
	{
	  for ( unsigned int j=0 ; j<nclus ; j++)
	    cluster_changed[j] = false;

	  nchanged = 0;
	  const PREC *px = X;
	  for ( unsigned int i=0 ; i<npts ; i++, px+=dim)
	    {
	      c[i] = assign_point_to_cluster(i, px, CX, dim, nclus, old_c[i], mindist, s, cl_dist, low_b);

#ifdef KMEANS_DEBUG
	      {
		/* If the assignments are not the same, there is still the BOUND_EPS difference 
		   which can be the reason of this*/
		unsigned int tmp = assign_point_to_cluster_ordinary(px, CX, dim, nclus);
		if (tmp != c[i])
		  {
		    printf("Found different cluster assignment.\n");
		    double d1 = compute_distance(px, CX+(tmp*dim), dim);
		    double d2 = compute_distance(px, CX+(c[i]*dim), dim);
		    assert( (d1>d2)?((d1-d2)<BOUND_EPS):((d2-d1)<BOUND_EPS) );
		  }
	      }
#endif

	      if (old_c[i] == c[i]) continue;
				
	      nchanged++;

	      cluster_changed[c[i]] = true;
	      cluster_changed[old_c[i]] = true;

	      remove_point_from_cluster(old_c[i], tCX, px, W[i], CN, CW, dim);
	      add_point_to_cluster(c[i], tCX, px, W[i], CN, CW, dim);
	    }

	}


      /* fill up empty clusters */
      for ( unsigned int j=0 ; j<nclus ; j++)
	{
	  if (CN[j]>0) continue; // DY: so j is an empty cluster
	  unsigned int *rperm = (unsigned int*)malloc(npts*sizeof(unsigned int));
	  if (rperm==NULL)	kmeans_error((char*)"Failed to allocate mem for permutation");

	  randperm(rperm, npts);
	  unsigned int i = 0; 
	  while (rperm[i]<npts && CN[c[rperm[i]]]<2) i++;
	  if (i==npts)continue;
	  i = rperm[i]; // DY: i is a point from a cluster with more than one point
#if KMEANS_VERBOSE>0
	  printf("empty cluster [%d], filling it with point [%d]\n", j, i);
#endif
	  cluster_changed[c[i]] = true; // DY: bug this should be c[i], we already did i=rperm[i]!
	  cluster_changed[j] = true;
	  const PREC *px = X + i*dim; // DY: px is the coordinates for the ith point
	  remove_point_from_cluster(c[i], tCX, px, W[i], CN, CW, dim);
	  c[i] = j;
	  add_point_to_cluster(j, tCX, px, W[i], CN, CW, dim);
	  /* void the bounds */
	  s[j] = (BOUND_PREC)0.0;
	  mindist[i] = 0.0;
	  if (use_low_b)
	    for ( unsigned int k=0 ; k<npts ; k++ )
	      low_b[k*nclus+j] = (BOUND_PREC)0.0;
			
	  nchanged++;
	  free(rperm);
	}

      /* no assignment changed: done */
      if (nchanged==0) break; 

      /* compute the offset */

      PREC *pcx = CX;
      PREC *tpcx = tCX;
      for ( unsigned int j=0 ; j<nclus ; j++, pcx+=dim, tpcx+=dim )
	{
	  offset[j] = (BOUND_PREC)0.0;
	  if (cluster_changed[j])
	    {
	      offset[j] = (BOUND_PREC)compute_distance(pcx, tpcx, dim);
	      memcpy(pcx, tpcx, dim*sizeof(PREC));
	    }
	}
		
      /* update the lower bound */
      if (use_low_b)
	{
	  for ( unsigned int i=0, cnt=0 ; i<npts ; i++ )
	    for ( unsigned int j=0 ; j<nclus ; j++, cnt++ )
	      {
		low_b[cnt] -= offset[j];
		if (low_b[cnt]<(BOUND_PREC)0.0) low_b[cnt] = (BOUND_PREC)0.0;
	      }
	}

      for ( unsigned int i=0; i<npts; i++)
	mindist[i] += (PREC)offset[c[i]];

      memcpy(old_c, c, npts*sizeof(unsigned int));

#if KMEANS_VERBOSE>0
      PREC rms = compute_rms(CX, X, W, c, dim, npts, nclus);
      fprintf(stderr, "iteration %4d, #(changed points): %4d, rms: %4.2f\n", (int)iteration, (int)nchanged, rms);
#endif

#if KMEANS_VERBOSE>1
      printf("saved at 2) %d\n", saved_two);
      printf("saved at 3i) %d\n", saved_three_one);
      printf("saved at 3ii) %d\n", saved_three_two);
      printf("saved at 3iii) %d\n", saved_three_three);
      printf("saved at 3b) %d\n", saved_three_b);
      saved_two=0;
      saved_three_one=0;
      saved_three_two=0;
      saved_three_three=0;
      saved_three_b=0;
#endif

      iteration++;

    }

#ifdef KMEANS_DEBUG
  for ( unsigned int j=0;j<nclus;j++)
    assert(CN[j]!=0); /* Empty cluster after all */
#endif


  /* find nearest cluster center if iteration reached maxiter */
  if (nchanged>0)
    {
      const PREC *px = X;
      for ( unsigned int i=0 ; i<npts ; i++, px+=dim)
	c[i] = assign_point_to_cluster_ordinary(px, CX, dim, nclus);
    }
  PREC rms = compute_rms(CX, X, W, c, dim, npts, nclus);

#if KMEANS_VERBOSE>0
    fprintf(stderr, "iteration %4d, #(changed points): %4d, rms: %f\n", (int)iteration, (int)nchanged, rms);
#endif

  if(low_b) free(low_b);
  free(cluster_changed);
  free(mindist);
  free(s);
  free(offset);
  free(cl_dist);
  free(tCX);
  free(CN);
  free(CW);
  free(old_c);

  return(rms);
}


/* Enis: my functions begins */
void furthest_first (double *CX, const double *X, unsigned int dim, unsigned int npts, unsigned int nclus)
{
  int max_i;
  double *distances, max_d;

  distances = (double*) malloc(npts * sizeof(*distances));
  for (int i = 0; i < npts; i++) distances[i] = 10e8;

  for (int i = 1; i < nclus; i++) {
    double *a = CX + (i - 1) * dim;
    max_d = 0;
    max_i = 0;
    for (int j = 0; j < npts; j++) {
      double d = compute_distance(a, X + j * dim, dim);
      if (d < distances[j])
	distances[j] = d;
      if (max_d < distances[j]) {
	max_d = distances[j];
	max_i = j;
      }
    }

    a = CX + i * dim;
    const double *b = X + max_i * dim;
    for (int j = 0; j < dim; j++) a[j] = b[j];
  }

  free(distances);
}

void furthest_first_sample (double *CX, const double *X, double *W, unsigned int dim, unsigned int npts, unsigned int nclus)
{
  int ind;
  double *distances, r;

  distances = (double*) malloc(npts * sizeof(*distances));

  for (int i = 0; i < npts; i++) distances[i] = PREC_MAX;

  for (int i = 1; i < nclus; i++) {
    double *a = CX + (i - 1) * dim;
    for (int j = 0; j < npts; j++) {
      double d = compute_distance(a, X + j * dim, dim);
      if (d < distances[j])
	distances[j] = d;
    }

    ind = 0;
    double sum = W[0] * distances[0] * distances[0];
    for (int j = 1; j < npts; j++) {
      double d = W[j] * distances[j] * distances[j];
      sum += d;
      r = ((double)rand() / RAND_MAX) * sum;
      if (r < d) ind = j;
    }

    a = CX + i * dim;
    const double *b = X + ind * dim;
    for (int j = 0; j < dim; j++) a[j] = b[j];
  }

  free(distances);
}

void rand_ff (double *CX, const double *X, unsigned int dim, unsigned int npts, unsigned int nclus)
{
  int r = rand() % npts;
  const double *a = X + r * dim;
  for (int i = 0; i < dim; i++) CX[i] = a[i];

  furthest_first(CX, X, dim, npts, nclus);
}

void kpp (double *CX, const double *X, double *W, unsigned int dim, unsigned int npts, unsigned int nclus)
{
  int r = rand() % npts;
  const double *a = X + r * dim;
  for (int i = 0; i < dim; i++) CX[i] = a[i];

  furthest_first_sample(CX, X, W, dim, npts, nclus);  
}

void random_init (double *CX, const double *X, unsigned int dim, unsigned int npts, unsigned int nclus)
{
  unsigned int *order = (unsigned int*)malloc(npts*sizeof(unsigned int));
  randperm(order, npts);
  for (unsigned int i=0; i<nclus; i++)
    for ( unsigned int k=0; k<dim; k++ )
      CX[(i*dim)+k] = X[order[i]*dim+k];
  free(order);
}
/* Enis: my functions ends */
