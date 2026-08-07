//  this file acts as a utility header for 3D geometry processing, point cloud manipulation, 
//  and spatial querying. It integrates standard math and vision libraries like Eigen (for linear algebra),
//  nanoflann (for fast nearest-neighbor searches), and Pangolin (for 3D vision and geometry)

// Copyright 2004-present Facebook. All Rights Reserved.
#include <vector>
// NB: This differs from the GitHub version due to the different location of the nanoflann header when installing from source
#include <nanoflann/nanoflann.hpp>
#include <pangolin/geometry/geometry.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>

// 1. KD-Tree Adaptor for Spatial Queries
// The first half of the code bridges a standard list of 3D points with the nanoflann library to enable rapid spatial searches 
// (e.g., finding the closest point to a given coordinate).

struct KdVertexList { // This struct serves as an adaptor for a list of 3D points to be used with the nanoflann KD-tree library.
 public:
  KdVertexList(const std::vector<Eigen::Vector3f>& points) : points_(points) {}

  inline size_t kdtree_get_point_count() const {
    return points_.size();
  }

  inline float kdtree_distance(const float* p1, const size_t idx_p2, size_t /*size*/) const {
    Eigen::Map<const Eigen::Vector3f> p(p1);
    return (p - points_[idx_p2]).squaredNorm();
  }

  inline float kdtree_get_pt(const size_t idx, int dim) const {
    return points_[idx](dim);
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const {
    return false;
  }

 private:
  std::vector<Eigen::Vector3f> points_;
};

using KdVertexListTree = nanoflann::KDTreeSingleIndexAdaptor< // This line defines a type alias for a KD-tree index adaptor that uses the L2 (Euclidean) distance metric,
    nanoflann::L2_Simple_Adaptor<float, KdVertexList>, // allowing for efficient nearest-neighbor searches in 3D space.
    KdVertexList,                                      // this line specifies that the KD-tree will be built from a KdVertexList, which is a wrapper around a vector of 3D points.
    3,                                                 // this line indicates that the KD-tree will operate in 3-dimensional space, as it is designed for 3D point clouds.
    int>;                                             // this line specifies that the index type for the KD-tree will be an integer, which is used to reference points in the KdVertexList.

// 2. 3D Geometry and Mesh Utilities
// The second half of the code declares a suite of utility functions 
// (whose implementations exist in an accompanying .cpp file). These are typically used in 3D computer vision pipelines:

std::vector<Eigen::Vector3f> EquiDistPointsOnSphere(const uint numSamples, const float radius); // Generates a set of points that are approximately equidistantly distributed on the surface of a sphere with a given radius.

std::vector<Eigen::Vector4f> ValidPointsFromIm(const pangolin::Image<Eigen::Vector4f>& verts); // Extracts valid 3D points from an image representation, filtering out any invalid or missing data.

std::vector<Eigen::Vector4f> ValidPointsAndTrisFromIm( // Extracts valid 3D points and their corresponding triangle indices from an image representation, filtering out any invalid or missing data.
    const pangolin::Image<Eigen::Vector4f>& pixNorms, // Image containing pixel normals
    std::vector<Eigen::Vector4f>& tris,              // Output vector to store valid triangle indices
    int& totalObs,                                  // Output parameter to store the total number of observations processed
    int& wrongObs);                                // Output parameter to store the number of observations that were deemed invalid or incorrect

float TriangleArea(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c); // Computes the area of a triangle defined by three 3D points (a, b, c) using the cross product method.

Eigen::Vector3f SamplePointFromTriangle( // Samples a random point from within a triangle defined by three 3D points (a, b, c) using barycentric coordinates.
    const Eigen::Vector3f& a,
    const Eigen::Vector3f& b,
    const Eigen::Vector3f& c);

std::pair<Eigen::Vector3f, float> ComputeNormalizationParameters( // Computes the normalization parameters (centroid and scale) for a given 3D geometry, which can be used to normalize the geometry to a standard size and position.
    pangolin::Geometry& geom,                                    // The geometry to be normalized
    const float buffer = 1.03);                                 // utilizing a slight buffer multiplier (defaulted to 1.03) to prevent the geometry from touching the exact edges of the bounding space

float BoundingCubeNormalization(      //
    pangolin::Geometry& geom,
    const bool fitToUnitSphere,
    const float buffer = 1.03);