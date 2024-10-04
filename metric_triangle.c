/******************************************************************/
//   FUNCTION Definition: xf_ShapeToMetric_Triangle
static int
xf_ShapeToMetric_Triangle(real *x, real *M, real *M_x)
{
/*
  PURPOSE: Computes element-implied metric for a triangle.
  INPUTS:  x : unrolled Q1 global coordinates of the nodes
  OUTPUTS: M : metric
           M_x : derivative of M w.r.t x (optional)
  RETURN:  Error Code
*/
  int ierr;
  int i, d, ip1, k, P[3];
  real A[9], b[3], e[6], *ei, A_xv[18];
  real v[3], rtemp[3];
  
  // make 3 edge vectors
  for (i=0; i<3; i++) 
    for (d=0, ip1=(i+1)%3; d<2; d++) 
      e[2*i+d] = x[ip1*2+d] - x[i*2+d];
  
  // require ei^T*M*ei = 1 for all edges i -> 3 x 3 system for entries in M
  for (k=0; k<9; k++) A[k] = 0.;
  for (i=0; i<3; i++){
    ei = e + 2*i; // ith edge vector
    A[3*i+0] =    ei[0]*ei[0]; // A{i0}
    A[3*i+1] = 2.*ei[0]*ei[1]; // A{i1}
    A[3*i+2] =    ei[1]*ei[1]; // A{i2}
    b[i] = 1.0;
  } // i

  // solve A*v = b
  xf_Call(xf_ComputePLU(A, 3, P));
  xf_Call(xf_SolvePLU(A, P, b, 3, v, rtemp));

  // set metric entries from v
  M[0] = v[0]; M[1] = M[2] = v[1]; M[3] = v[2];

  return xf_OK;  
}

