#ifndef _DENSE_INFERENCE_H
#define _DENSE_INFERENCE_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <boost/python.hpp>

struct InputData3D {
  int MaxIterations;
  float PosXStd;
  float PosYStd;
  float PosZStd;
  float PosW;
  float BilateralXStd;
  float BilateralYStd;
  float BilateralZStd;
  float BilateralGStd; // gray
  float BilateralW;

  void print_settings(void);
};

class DenseCRF3DProcessor {
protected:
  // Data to initialize CRF
  InputData3D inp_;

  // verbose
  bool verb_ = false;

  float *unary_3d_ = NULL;
  float *feat_ = NULL;
  float *img_ = NULL;
  short *map_ = NULL;

  int feat_row_, feat_col_, feat_slc_, feat_channel_;
  int img_row_, img_col_, img_slc_;

  // vector to python list helper (by Markus Rempfler)
  PyObject* stdVecToPyListShort(const std::vector<short>& vec);
  PyObject* stdVecToPyListFloat(const std::vector<float>& vec);

  // unary helper
  void calc_unary(float * unary, float * feat, int feat_row, int feat_col, int feat_slc, int feat_channel);

  // tidy up
  void tidy_up(void);

public:
  // initialize with standard values
  DenseCRF3DProcessor();
  // initialize with custom values
  DenseCRF3DProcessor(int MaxIterations, float PosXStd, float PosYStd, float PosZStd, float PosW, float BilateralXStd, float BilateralYStd, float BilateralZStd, float BilateralGStd, float BilateralW, bool verbose);
  ~DenseCRF3DProcessor();

  // Due to a huge variance in number of z-slices, a setter for 'PosZStd'. Standard value of PosZStd should be proportional to number of slices
  void set_pos_z_std(float PosZStd);
  void set_bi_z_std(float BilateralZStd);

  // set feature space from numpy array
  void set_feature(boost::python::numeric::array feature, int feat_row, int feat_col, int feat_slc, int feat_channel, const std::string &numpy_data_type);
  PyObject* get_feature();

  // set image volume
  void set_image(boost::python::numeric::array image, int img_row, int img_col, int img_slc, const std::string &numpy_data_type);
  PyObject* get_image();

  // here the actual calculation happens
  void calculate_map(std::vector<short> &v_map);

  // anticipate required memory
  static long get_memory_requirements(const int W, const int H, const int D, const int C);

  // wrapper for python to return the calculated map as numpy array
  PyObject* calc_res_array(void);
};

#endif


