// **************************************************************************
//
//    PARALUTION   www.paralution.com
//
//    Copyright (C) 2015  PARALUTION Labs UG (haftungsbeschränkt) & Co. KG
//                        Am Hasensprung 6, 76571 Gaggenau
//                        Handelsregister: Amtsgericht Mannheim, HRA 706051
//                        Vertreten durch:
//                        PARALUTION Labs Verwaltungs UG (haftungsbeschränkt)
//                        Am Hasensprung 6, 76571 Gaggenau
//                        Handelsregister: Amtsgericht Mannheim, HRB 721277
//                        Geschäftsführer: Dimitar Lukarski, Nico Trost
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// **************************************************************************



// PARALUTION version 1.1.0 


#ifndef PARALUTION_GPU_CUSPARSE_CSR_HPP_
#define PARALUTION_GPU_CUSPARSE_CSR_HPP_

namespace paralution {

hipsparseStatus_t __cusparseXcsrgeam__(hipsparseHandle_t handle, int m, int n,
                                      const double *alpha,
                                      const hipsparseMatDescr_t descrA, int nnzA,
                                      const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                      const double *beta,
                                      const hipsparseMatDescr_t descrB, int nnzB,
                                      const double *csrValB, const int *csrRowPtrB, const int *csrColIndB,
                                      const hipsparseMatDescr_t descrC,
                                      double *csrValC, int *csrRowPtrC, int *csrColIndC) {

  return cusparseDcsrgeam(handle, 
                          m, n,
                          alpha,
                          descrA, nnzA,
                          csrValA, csrRowPtrA, csrColIndA,
                          beta,
                          descrB, nnzB,
                          csrValB, csrRowPtrB, csrColIndB,
                          descrC,
                          csrValC, csrRowPtrC, csrColIndC);
}

hipsparseStatus_t __cusparseXcsrgeam__(hipsparseHandle_t handle, int m, int n,
                                      const float *alpha,
                                      const hipsparseMatDescr_t descrA, int nnzA,
                                      const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                      const float *beta,
                                      const hipsparseMatDescr_t descrB, int nnzB,
                                      const float *csrValB, const int *csrRowPtrB, const int *csrColIndB,
                                      const hipsparseMatDescr_t descrC,
                                      float *csrValC, int *csrRowPtrC, int *csrColIndC) {
  return cusparseScsrgeam(handle, 
                          m, n,
                          alpha,
                          descrA, nnzA,
                          csrValA, csrRowPtrA, csrColIndA,
                          beta,
                          descrB, nnzB,
                          csrValB, csrRowPtrB, csrColIndB,
                          descrC,
                          csrValC, csrRowPtrC, csrColIndC);

}

hipsparseStatus_t  __cusparseXcsrgemm__(hipsparseHandle_t handle,
                                       hipsparseOperation_t transA, hipsparseOperation_t transB,
                                       int m, int n, int k,
                                       const hipsparseMatDescr_t descrA, const int nnzA,
                                       const double *csrValA,
                                       const int *csrRowPtrA, const int *csrColIndA,
                                       const hipsparseMatDescr_t descrB, const int nnzB,                            
                                       const double *csrValB, 
                                       const int *csrRowPtrB, const int *csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       double *csrValC,
                                       const int *csrRowPtrC, int *csrColIndC ) {
  
  return hipsparseDcsrgemm(handle,
                          transA, transB,
                          m, n, k,
                          descrA, nnzA,
                          csrValA,
                          csrRowPtrA, csrColIndA,
                          descrB, nnzB,                            
                          csrValB, 
                          csrRowPtrB, csrColIndB,
                          descrC,
                          csrValC,
                          csrRowPtrC, csrColIndC );

}

hipsparseStatus_t  __cusparseXcsrgemm__(hipsparseHandle_t handle,
                                       hipsparseOperation_t transA, hipsparseOperation_t transB,
                                       int m, int n, int k,
                                       const hipsparseMatDescr_t descrA, const int nnzA,
                                       const float *csrValA,
                                       const int *csrRowPtrA, const int *csrColIndA,
                                       const hipsparseMatDescr_t descrB, const int nnzB,                            
                                       const float *csrValB, 
                                       const int *csrRowPtrB, const int *csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       float *csrValC,
                                       const int *csrRowPtrC, int *csrColIndC ) {

  return hipsparseScsrgemm(handle,
                          transA, transB,
                          m, n, k,
                          descrA, nnzA,
                          csrValA,
                          csrRowPtrA, csrColIndA,
                          descrB, nnzB,                            
                          csrValB, 
                          csrRowPtrB, csrColIndB,
                          descrC,
                          csrValC,
                          csrRowPtrC, csrColIndC );

}


}

#endif
