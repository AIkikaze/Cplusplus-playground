#ifndef LINE2DUP_PRECOMP_HPP
#define LINE2DUP_PRECOMP_HPP

#include "../../../MIPP/mipp.h"
#include <omp.h>

#include <set>
#include <map>
#include <list>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define line2d_eps 1e-3f 
#define _degree_(x) ((x)*CV_PI) / 180.0

#define QUANTIZE_BASE 16

#if QUANTIZE_BASE == 16
#define QUANTIZE_TYPE CV_16U
typedef ushort quantize_type;
#elif QUNATIZE_BASE == 8
#define QUANTIZE_TYPE CV_8U
typedef uchar quantize_type
#endif

#endif // LINE2DUP_PRECOMP_HPP