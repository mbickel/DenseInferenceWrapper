/*
*
* Python wrapper for DenseCRF
*
* by Marc Bickel (2015)
*
* Python wrapper for 2D-denseCRF
*
* by Wendong Xu (2018)
*
*/

#include <stdlib.h>

#include <Python.h>

#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include "boost/python/tuple.hpp"
#include <numpy/ndarrayobject.h>

#include <iostream>
#include <vector>
#include <exception>

#include "../libDenseCRF/densecrf.h"
#include "../libDenseCRF/util.h"
#include "dense_inference.h"

using namespace boost::python;

//////////////////////////////////
/////  Initialize / Dealloc  /////
//////////////////////////////////
DenseCRF3DProcessor::DenseCRF3DProcessor() {
	feat_row_ = 0, feat_col_ = 0, feat_slc_ = 0, feat_channel_ = 0;
	img_row_ = 0, img_col_ = 0, img_slc_ = 0;

	inp_.MaxIterations = 10;
	inp_.PosXStd = 3;
	inp_.PosYStd = 3;
	inp_.PosZStd = 3;
	inp_.PosW = 3;
	inp_.BilateralXStd = 60;
	inp_.BilateralYStd = 60;
	inp_.BilateralZStd = 60;
	inp_.BilateralGStd = 20;
	inp_.BilateralW = 10;

	verb_ = false;
}
DenseCRF3DProcessor::DenseCRF3DProcessor(int MaxIterations, float PosXStd, float PosYStd, float PosZStd, float PosW, float BilateralXStd, float BilateralYStd, float BilateralZStd, float BilateralGStd, float BilateralW, bool verbose) {
	feat_row_ = 0, feat_col_ = 0, feat_slc_ = 0, feat_channel_ = 0;
	img_row_ = 0, img_col_ = 0, img_slc_ = 0;

	inp_.MaxIterations = MaxIterations;
	inp_.PosXStd = PosXStd;
	inp_.PosYStd = PosYStd;
	inp_.PosZStd = PosZStd;
	inp_.PosW = PosW;
	inp_.BilateralXStd = BilateralXStd;
	inp_.BilateralYStd = BilateralYStd;
	inp_.BilateralZStd = BilateralZStd;
	inp_.BilateralGStd = BilateralGStd;
	inp_.BilateralW = BilateralW;

	verb_ = verbose;
}
void DenseCRF3DProcessor::tidy_up(void) {
	if (unary_3d_ != NULL) {
		delete[] unary_3d_;
		unary_3d_ = NULL;
	}

	if (feat_ != NULL) {
		delete[] feat_;
		feat_ = NULL;
	}

	if (img_ != NULL) {
		delete[] img_;
		img_ = NULL;
	}

	if (map_ != NULL) {
		delete[] map_;
		map_ = NULL;
	}

	// set shape variables to 0
	feat_row_ = 0, feat_col_ = 0, feat_slc_ = 0, feat_channel_ = 0;
	img_row_ = 0, img_col_ = 0, img_slc_ = 0;
}
DenseCRF3DProcessor::~DenseCRF3DProcessor() {
	tidy_up();
}

////////////////////////////////////
///// Protected Helper Methods /////
////////////////////////////////////
PyObject* DenseCRF3DProcessor::stdVecToPyListFloat(const std::vector<float>& vec) {
	boost::python::list* l = new boost::python::list();
	for (size_t i = 0; i < vec.size(); i++)
		(*l).append(vec[i]);

	return l->ptr();
}
PyObject* DenseCRF3DProcessor::stdVecToPyListShort(const std::vector<short>& vec) {
	boost::python::list* l = new boost::python::list();
	for (size_t i = 0; i < vec.size(); i++)
		(*l).append(vec[i]);

	return l->ptr();
}
void DenseCRF3DProcessor::calc_unary(float * unary, float * feat, int feat_row, int feat_col, int feat_slc, int feat_channel) {
	// calc a -log() over all entries
	for (long i = 0; i < (feat_row * feat_col * feat_slc * feat_channel); i++) {
		unary[i] = -log(feat[i]);
	}
}

////////////////////////////////////////////////////////////////
///// Image and Feature Volume (and BilateralZStd) Setters /////
////////////////////////////////////////////////////////////////
void DenseCRF3DProcessor::set_feature(boost::python::numeric::array feature, int feat_row, int feat_col, int feat_slc, int feat_channel, const std::string &numpy_data_type) {
	// first tidy up
	if (feat_ != NULL)
		delete[] feat_;

	if (numpy_data_type == "float64") {
		boost::python::numeric::array in = feature;

		// set shape
		feat_row_ = feat_row;
		feat_col_ = feat_col;
		feat_slc_ = feat_slc;
		feat_channel_ = feat_channel;

		// allocate memory
		feat_ = new float[feat_row * feat_col * feat_slc * feat_channel];

		// iterate over numpy array, extract and cast to float
		for (int k = 0; k<feat_slc; k++)
			for (int j = 0; j<feat_col; j++)
				for (int i = 0; i<feat_row; i++)
					for (int c = 0; c<feat_channel; c++) {
						feat_[(k*feat_col*feat_row + j * feat_row + i)*feat_channel + c] = (float)extract<double>(in[make_tuple(i, j, k, c)]);
					}
	}
	else {
		throw std::invalid_argument("Invalid data type: " + numpy_data_type + ". Abort.");
	}
}
PyObject* DenseCRF3DProcessor::get_feature() {
	// transform to vector
	std::vector<float> v;
	float * start = feat_;
	float * end = feat_ + feat_row_ * feat_col_ * feat_slc_ * feat_channel_;
	v.clear();
	v.insert(v.end(), start, end);

	return stdVecToPyListFloat(v);
}
// image packing order is std fortran order, like numpy.ravel(img, order='F'), expects image with dynamic range [0, 1]
void DenseCRF3DProcessor::set_image(boost::python::numeric::array image, int img_row, int img_col, int img_slc, const std::string &numpy_data_type) {
	// first tidy up
	if (img_ != NULL)
		delete[] img_;

	if (numpy_data_type == "float64") {
		boost::python::numeric::array in = image;

		// set shape
		img_row_ = img_row;
		img_col_ = img_col;
		img_slc_ = img_slc;

		// allocate memory
		img_ = new float[img_row * img_col * img_slc];

		// iterate over numpy array, extract and cast to float
		for (int k = 0; k<img_slc; k++)
			for (int j = 0; j<img_col; j++)
				for (int i = 0; i<img_row; i++) {
					img_[(k * img_col * img_row + j * img_row + i)] = ((float)extract<double>(in[make_tuple(i, j, k)])) * 255.0f;
				}
	}
	else {
		throw std::invalid_argument("Invalid data type: " + numpy_data_type + ". Abort.");
	}
}
PyObject* DenseCRF3DProcessor::get_image() {
	// transform to vector
	std::vector<float> v;
	float * start = img_;
	float * end = img_ + img_row_ * img_col_ * img_slc_;
	v.clear();
	v.insert(v.end(), start, end);

	return stdVecToPyListFloat(v);
}
void DenseCRF3DProcessor::set_pos_z_std(float PosZStd) {
	inp_.PosZStd = PosZStd;
}
void DenseCRF3DProcessor::set_bi_z_std(float BilateralZStd) {
	inp_.BilateralZStd = BilateralZStd;
}
/////////////////////////
///// CRF Processor /////
/////////////////////////
void DenseCRF3DProcessor::calculate_map(std::vector<short> &v_map) {
	// check if image and features are set
	if (img_ == NULL || feat_ == NULL) {
		std::cerr << "Error. Image or features not set. Abort." << std::endl;
		return;
	}

	// check if shapes of image and features are equal
	if ((img_row_ != feat_row_) || (img_col_ != feat_col_) || (img_slc_ != feat_slc_)) {
		std::cerr << "Error. Image shape differs from features shape. Abort." << std::endl;
		return;
	}

	// check if shape not 0
	if ((!img_row_) || (!img_col_) || (!img_slc_)) {
		std::cerr << "Error. Either rows, columns or slices are 0. Abort." << std::endl;
		return;
	}
	// transform feature to unary
	unary_3d_ = new float[feat_row_ * feat_col_ * feat_slc_ * feat_channel_];
	calc_unary(unary_3d_, feat_, feat_row_, feat_col_, feat_slc_, feat_channel_);

	// Initialize the magic..
	DenseCRF3D crf(img_slc_, img_col_, img_row_, feat_channel_);

	// Set verbose
	crf.verbose(verb_);

	// Specify the unary potential as an array of size W*H*D*(#classes)
	// packing order: x0y0z0l0 x0y0z0l1 x0y0z0l2 .. x1y0z0l0 x1y0z0l1 ... (row-order) (image packing order is std fortran order, like numpy.ravel(img, order='F'))
	if (crf.isVerbose())
		std::cout << "pre energy" << std::endl;
	crf.setUnaryEnergy(unary_3d_);
	// add a gray intensity independent term (feature = pixel location 0..W-1, 0..H-1)
	if (crf.isVerbose())
		std::cout << "pre gaussian" << std::endl;
	crf.addPairwiseGaussian(inp_.PosXStd, inp_.PosYStd, inp_.PosZStd, inp_.PosW);
	// add a gray intensity dependent term (feature = xyzg)
	if (crf.isVerbose())
		std::cout << "pre bilateral" << std::endl;
	crf.addPairwiseBilateral(inp_.BilateralXStd, inp_.BilateralYStd, inp_.BilateralZStd, inp_.BilateralGStd, img_, inp_.BilateralW);

	// create map (=maximum a posteriori)
	map_ = new short[img_row_ * img_col_ * img_slc_];

	if (crf.isVerbose())
		std::cout << "pre premap" << std::endl;

	crf.map(inp_.MaxIterations, map_);

	// print out some values from raw Map array configuration
	if (crf.isVerbose()) {
		long ctr = 0;
		for (long i = 0; i < img_row_ * img_col_ * img_slc_; i++) {
			if (map_[i] > 0) {
				ctr++;
			}
			if (i % (img_row_ * img_col_ * img_slc_ / 20) == 0) {
				std::cout << map_[i] << " ";
			}
		}
		std::cout << std::endl << "Maximum a posteriori > 0: " << ctr << std::endl;
	}

	// transform to vector
	short * start = map_;
	short * end = map_ + img_row_ * img_col_ * img_slc_;
	v_map.clear();
	v_map.insert(v_map.end(), start, end);

	// tidy up
	tidy_up();

	if (crf.isVerbose())
		std::cout << "Done." << std::endl;
}

/////////////////////////
///// Public Helper /////
/////////////////////////
long DenseCRF3DProcessor::get_memory_requirements(const int W, const int H, const int D, const int C) {
	long M = (long)C;
	long N = (long)(W*H*D);

	long FL = (long)sizeof(float);
	long SH = (long)sizeof(short);

	long dCRF = 6 * N * M * FL;
	long d3CRF = (2 * N * M + N) * FL + (N * M * SH);
	long bil = 6 * N;

	return dCRF + d3CRF + bil;
}


//////////////////////////
///// Python Wrapper /////
//////////////////////////
PyObject* DenseCRF3DProcessor::calc_res_array() {
	std::vector<short> map;
	calculate_map(map);

	// transform to numpy array and return it
	if (map.size() > 0) {
		return stdVecToPyListShort(map);
	}
	else {
		throw std::runtime_error("Invalid map. Calculate_map() did not return successfully.");
	}
}



////////////////////////////////////////
/////  2D-denseCRF python wrapper  /////
/////  Implemented by Wendong Xu   /////
////////////////////////////////////////
DenseCRF2DProcessor::DenseCRF2DProcessor() {
	feat_row_ = 0, feat_col_ = 0, feat_channel_ = 0;
	img_row_ = 0, img_col_ = 0;

	inp_.MaxIterations = 10;
	inp_.PosXStd = 3;
	inp_.PosYStd = 3;
	inp_.PosW = 3;
	inp_.BilateralXStd = 60;
	inp_.BilateralYStd = 60;
	inp_.BilateralGStd = 20;
	inp_.BilateralW = 10;

	verb_ = false;
}

DenseCRF2DProcessor::DenseCRF2DProcessor(int MaxIterations, float PosXStd, float PosYStd, float PosW, float BilateralXStd, float BilateralYStd, float BilateralGStd, float BilateralW, bool verbose) {
	feat_row_ = 0, feat_col_ = 0, feat_channel_ = 0;
	img_row_ = 0, img_col_ = 0;

	inp_.MaxIterations = MaxIterations;
	inp_.PosXStd = PosXStd;
	inp_.PosYStd = PosYStd;
	inp_.PosW = PosW;
	inp_.BilateralXStd = BilateralXStd;
	inp_.BilateralYStd = BilateralYStd;
	inp_.BilateralGStd = BilateralGStd;
	inp_.BilateralW = BilateralW;

	verb_ = verbose;
}

void DenseCRF2DProcessor::tidy_up(void) {
	if (unary_2d_ != NULL) {
		delete[] unary_2d_;
		unary_2d_ = NULL;
	}

	if (feat_ != NULL) {
		delete[] feat_;
		feat_ = NULL;
	}

	if (img_ != NULL) {
		delete[] img_;
		img_ = NULL;
	}

	if (map_ != NULL) {
		delete[] map_;
		map_ = NULL;
	}

	// set shape variables to 0
	feat_row_ = 0, feat_col_ = 0, feat_channel_ = 0;
	img_row_ = 0, img_col_ = 0;
}

DenseCRF2DProcessor::~DenseCRF2DProcessor() {
	tidy_up();
}

PyObject* DenseCRF2DProcessor::stdVecToPyListShort(const std::vector<short>& vec) {
	boost::python::list* l = new boost::python::list();
	for (size_t i = 0; i < vec.size(); i++)
		(*l).append(vec[i]);

	return l->ptr();
}

PyObject* DenseCRF2DProcessor::stdVecToPyListFloat(const std::vector<float>& vec) {
	boost::python::list* l = new boost::python::list();
	for (size_t i = 0; i < vec.size(); i++)
		(*l).append(vec[i]);

	return l->ptr();
}

void DenseCRF2DProcessor::calc_unary(float * unary, float * feat, int feat_row, int feat_col, int feat_channel) {
	for (long i = 0; i < (feat_row * feat_col * feat_channel); i++) {
		unary[i] = -log(feat[i]);
	}
}

void DenseCRF2DProcessor::set_feature(boost::python::numeric::array feature, int feat_row, int feat_col, int feat_channel, const std::string &numpy_data_type) {
	if (feat_ != NULL) {
		delete[]feat_;
	}
	if (numpy_data_type == "float64") {
		boost::python::numeric::array in = feature;
		feat_row_ = feat_row;
		feat_col_ = feat_col;
		feat_channel_ = feat_channel;
		feat_ = new float[feat_row * feat_col * feat_channel];
		for (int j = 0; j < feat_col; j++) {
			for (int i = 0; i < feat_row; i++) {
				for (int c = 0; c < feat_channel; c++) {
					feat_[(j * feat_row + i) * feat_channel + c] = (float)extract<double>(in[make_tuple(i, j, c)]);
				}
			}
		}
	}
	else {
		throw std::invalid_argument("Invalid data type: " + numpy_data_type + ". Abort.");
	}
}

PyObject* DenseCRF2DProcessor::get_feature() {
	std::vector<float> v;
	float * start = feat_;
	float * end = feat_ + feat_row_ * feat_col_ * feat_channel_;
	v.clear();
	v.insert(v.end(), start, end);
	return stdVecToPyListFloat(v);
}

void DenseCRF2DProcessor::set_image(boost::python::numeric::array image, int img_row, int img_col, const std::string &numpy_data_type) {
	if (img_ != NULL) {
		delete[] img_;
	}
	if (numpy_data_type == "float64") {
		boost::python::numeric::array in = image;
		img_row_ = img_row;
		img_col_ = img_col;
		img_ = new float[img_row * img_col];
		for (int j = 0; j < img_col; j++) {
			for (int i = 0; i < img_row; i++) {
				img_[j * img_row + i] = ((float)extract<double>(in[make_tuple(i, j)])) * 255.0f;
			}
		}
	}
	else {
		throw std::invalid_argument("Invalid data type: " + numpy_data_type + ". Abort.");
	}
}

PyObject* DenseCRF2DProcessor::get_image() {
	std::vector<float> v;
	float * start = img_;
	float * end = img_ + img_row_ * img_col_;
	v.clear();
	v.insert(v.end(), start, end);

	return stdVecToPyListFloat(v);
}

void DenseCRF2DProcessor::calculate_map(std::vector<short> &v_map) {
	if (img_ == NULL || feat_ == NULL) {
		std::cerr << "Error. Image or features not set. Abort." << std::endl;
		return;
	}
	if ((img_row_ != feat_row_) || (img_col_ != feat_col_)) {
		std::cerr << "Error. Image shape differs from features shape. Abort." << std::endl;
		return;
	}
	if ((!img_row_) || (!img_col_)) {
		std::cerr << "Error. Either rows, columns or slices are 0. Abort." << std::endl;
		return;
	}
	unary_2d_ = new float[feat_row_ * feat_col_ * feat_channel_];
	calc_unary(unary_2d_, feat_, feat_row_, feat_col_, feat_channel_);
	DenseCRF2D crf(img_col_, img_row_, feat_channel_);
	crf.verbose(verb_);
	if (crf.isVerbose()) {
		std::cout << "pre energy" << std::endl;
	}
	crf.setUnaryEnergy(unary_2d_);
	if (crf.isVerbose()) {
		std::cout << "pre gaussian" << std::endl;
	}
	crf.addPairwiseGaussian(inp_.PosXStd, inp_.PosYStd, inp_.PosW);
	if (crf.isVerbose()) {
		std::cout << "pre bilateral" << std::endl;
	}
	map_ = new short[img_row_ * img_col_];
	if (crf.isVerbose()) {
		std::cout << "pre premap" << std::endl;
	}
	crf.map(inp_.MaxIterations, map_);
	if (crf.isVerbose()) {
		long ctr = 0;
		for (long i = 0; i < img_row_ * img_col_; i++) {
			if (map_[i] > 0) {
				ctr++;
			}
			if (i % (img_row_ * img_col_ / 20) == 0) {
				std::cout << map_[i] << " ";
			}
		}
		std::cout << std::endl << "Maximum a posteriori > 0: " << ctr << std::endl;
	}
	short * start = map_;
	short * end = map_ + img_row_ * img_col_;
	v_map.clear();
	v_map.insert(v_map.end(), start, end);
	tidy_up();
	if (crf.isVerbose()) {
		std::cout << "Done." << std::endl;
	}
}

long DenseCRF2DProcessor::get_memory_requirements(const int W, const int H, const int C) {
	long M = (long)C;
	long N = (long)(W*H);

	long FL = (long)sizeof(float);
	long SH = (long)sizeof(short);

	long dCRF = 6 * N * M * FL;
	long d3CRF = (2 * N * M + N) * FL + (N * M * SH);
	long bil = 6 * N;

	return dCRF + d3CRF + bil;
}

PyObject* DenseCRF2DProcessor::calc_res_array() {
	std::vector<short> map;
	calculate_map(map);
	if (map.size() > 0) {
		return stdVecToPyListShort(map);
	}
	else {
		throw std::runtime_error("Invalid map. Calculate_map() did not return successfully.");
	}
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(dense_inference) {
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	class_<DenseCRF3DProcessor>("DenseCRF3DProcessor", init<>())
		.def(init<int, float, float, float, float, float, float, float, float, float, bool>())
		.def("set_pos_z_std", &DenseCRF3DProcessor::set_pos_z_std)
		.def("set_bi_z_std", &DenseCRF3DProcessor::set_bi_z_std)
		.def("set_feature", &DenseCRF3DProcessor::set_feature)
		.def("get_feature", &DenseCRF3DProcessor::get_feature)
		.def("set_image", &DenseCRF3DProcessor::set_image)
		.def("get_image", &DenseCRF3DProcessor::get_image)
		.def("get_memory_requirements", &DenseCRF3DProcessor::get_memory_requirements)
		.staticmethod("get_memory_requirements")
		.def("calc_res_array", &DenseCRF3DProcessor::calc_res_array)
		;

	class_<DenseCRF2DProcessor>("DenseCRF2DProcessor", init<>())
		.def(init<int, float, float, float, float, float, float, float, bool>())
		.def("set_feature", &DenseCRF2DProcessor::set_feature)
		.def("get_feature", &DenseCRF2DProcessor::get_feature)
		.def("set_image", &DenseCRF2DProcessor::set_image)
		.def("get_image", &DenseCRF2DProcessor::get_image)
		.def("get_memory_requirements", &DenseCRF2DProcessor::get_memory_requirements)
		.staticmethod("get_memory_requirements")
		.def("calc_res_array", &DenseCRF2DProcessor::calc_res_array)
		;
}
