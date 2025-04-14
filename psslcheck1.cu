#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <cuda_runtime.h>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include "segment.h"
#include "readData.h"
#include "sortSweep.h"

using namespace std;

// Helper function for CUDA error checking
inline void cudaCheckError(cudaError_t error) {
    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(error) << endl;
        exit(1);
    }
}

struct BBoxWithRegionID {
    float x, y, z, w;
    int regionID;
    __host__ __device__
    BBoxWithRegionID(float _x = 0, float _y = 0, float _z = 0, float _w = 0, int _regionID = -1)
        : x(_x), y(_y), z(_z), w(_w), regionID(_regionID) {}
};

struct Segment {
    double dx;
    double dy;
    double sx;
    double sy;
    int regionID;  // in the original region.
    int segmentID;     // store the original index of this segment
    int interiorAbove;
    int interiorBelow;
    int fileID;     // 0 for file1, 1 for file2
    
    // Add constructor to match the parameters passed to push_back
    Segment(double _dx, double _dy, double _sx, double _sy, 
            int _regionID, int _segmentID, 
            int _interiorAbove, int _interiorBelow, int _fileID)
        : dx(_dx), dy(_dy), sx(_sx), sy(_sy), 
          regionID(_regionID), segmentID(_segmentID), 
          interiorAbove(_interiorAbove), interiorBelow(_interiorBelow),
          fileID(_fileID) {}
};

struct comparisonFunctor {
    BBoxWithRegionID* bboxes1;
    BBoxWithRegionID* bboxes2;
    thrust::pair<int, int>* intersecting_pairs;
    int* counter;
    int max_pairs;

    __host__ __device__
    comparisonFunctor(BBoxWithRegionID* _bboxes1, BBoxWithRegionID* _bboxes2,
                    thrust::pair<int, int>* _intersecting_pairs,
                    int* _counter, int _max_pairs)
        : bboxes1(_bboxes1), bboxes2(_bboxes2),
        intersecting_pairs(_intersecting_pairs), counter(_counter), max_pairs(_max_pairs) {}

    __device__
    void operator()(int i, int j) const {
        const BBoxWithRegionID& r1 = bboxes1[i];
        const BBoxWithRegionID& r2 = bboxes2[j];

        if (!(r1.x > r2.z || r2.x > r1.z || r1.y > r2.w || r2.y > r1.w)) {
            int pair_idx = atomicAdd(counter, 1);
            if (pair_idx < max_pairs) {
                intersecting_pairs[pair_idx] = thrust::pair<int, int>(i, j);
            }
        }
    }
};

struct LaunchFunctor {
    BBoxWithRegionID* bboxes1;
    BBoxWithRegionID* bboxes2;
    int bboxes1_size;
    int bboxes2_size;
    thrust::pair<int, int>* intersecting_pairs;
    int* counter;
    int max_pairs;

    __host__ __device__
    LaunchFunctor(BBoxWithRegionID* _bboxes1, BBoxWithRegionID* _bboxes2,
                int _bboxes1_size, int _bboxes2_size,
                thrust::pair<int, int>* _intersecting_pairs,
                int* _counter, int _max_pairs)
        : bboxes1(_bboxes1), bboxes2(_bboxes2),
        bboxes1_size(_bboxes1_size), bboxes2_size(_bboxes2_size),
        intersecting_pairs(_intersecting_pairs), counter(_counter), max_pairs(_max_pairs) {}

    __device__
    void operator()(int i) const {
        for (int j = 0; j < bboxes2_size; ++j) {
            comparisonFunctor(bboxes1, bboxes2, intersecting_pairs, counter, max_pairs)(i, j);
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " [file1.dxf] [file2.dxf]" << endl;
        return 1;
    }

    string file1(argv[1]), file2(argv[2]);
    segments regions1, regions2;
    readData(file1, regions1);
    readData(file2, regions2);

    if (regions1.numRegions == 0 || regions2.numRegions == 0) {
        cerr << "Nothing read from one of the files" << endl;
        return 1;
    }

    cerr << "numRegions 1 and 2: " << regions1.numRegions << " " << regions2.numRegions << endl;

    regions1.computeRegionStartIndex();
    regions2.computeRegionStartIndex();
    regions1.computeGlobalBBox();
    regions2.computeGlobalBBox();
    // regions1.shiftToQuadrant1();
    // regions2.shiftToQuadrant1();
    regions1.computeBBoxes();
    regions2.computeBBoxes();

    // Build bounding box arrays
   // In the bounding box creation part:
// Build bounding box arrays - use the same regionID as in segments

    auto start = std::chrono::high_resolution_clock::now();
    thrust::host_vector<BBoxWithRegionID> bboxes1, bboxes2;
    for (size_t i = 0; i < regions1.numRegions; ++i) {
        // Find the first segment with this regionID to get the correct ID
        int first_seg = regions1.regionStartIndex[i];
        int actual_regionID = regions1.regionID[first_seg];
        
        bboxes1.push_back(BBoxWithRegionID(regions1.lx[i], regions1.ly[i],
                                        regions1.ux[i], regions1.uy[i],
                                        actual_regionID)); // Use the segment's regionID
    }

    for (size_t j = 0; j < regions2.numRegions; ++j) {
        int first_seg = regions2.regionStartIndex[j];
        int actual_regionID = regions2.regionID[first_seg];
        
        bboxes2.push_back(BBoxWithRegionID(regions2.lx[j], regions2.ly[j],
                                        regions2.ux[j], regions2.uy[j],
                                        actual_regionID)); // Use the segment's regionID
    }

    int bboxes1_size = bboxes1.size();
    int bboxes2_size = bboxes2.size();

    // Copy bounding boxes to device
    thrust::device_vector<BBoxWithRegionID> d_bboxes1 = bboxes1;
    thrust::device_vector<BBoxWithRegionID> d_bboxes2 = bboxes2;
    BBoxWithRegionID* raw_d_bboxes1 = thrust::raw_pointer_cast(d_bboxes1.data());
    BBoxWithRegionID* raw_d_bboxes2 = thrust::raw_pointer_cast(d_bboxes2.data());

    int max_pairs = max(bboxes1_size * bboxes2_size, 10000000);
    thrust::device_vector<thrust::pair<int, int>> intersecting_pairs(max_pairs);
    thrust::pair<int, int>* raw_intersecting_pairs = thrust::raw_pointer_cast(intersecting_pairs.data());

    int* d_counter;
    cudaCheckError(cudaMalloc(&d_counter, sizeof(int)));
    cudaCheckError(cudaMemset(d_counter, 0, sizeof(int)));

    

    thrust::for_each(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(bboxes1_size),
                    LaunchFunctor(raw_d_bboxes1, raw_d_bboxes2, bboxes1_size, bboxes2_size,
                                raw_intersecting_pairs, d_counter, max_pairs));

    int num_intersections;
    cudaCheckError(cudaMemcpy(&num_intersections, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    cout << "Number of intersecting bounding-box pairs: " << num_intersections << endl;

    auto end = std::chrono::high_resolution_clock::now();

    thrust::host_vector<thrust::pair<int, int>> host_intersecting_pairs(num_intersections);
    thrust::copy(intersecting_pairs.begin(), intersecting_pairs.begin() + num_intersections,
                host_intersecting_pairs.begin());

    cudaFree(d_counter);

    // Then later, when collecting segments:
    unordered_set<int> unique_regions1, unique_regions2;
    for (int i = 0; i < num_intersections; ++i) {
        // Get the actual regionIDs from the bounding boxes
        unique_regions1.insert(bboxes1[host_intersecting_pairs[i].first].regionID);
        unique_regions2.insert(bboxes2[host_intersecting_pairs[i].second].regionID);
    }

    cout << "Number of unique regions from file1: " << unique_regions1.size() << endl;
    cout << "Number of unique regions from file2: " << unique_regions2.size() << endl;

    // Collect all segments from intersecting regions
    vector<Segment> all_segments;
    all_segments.reserve(regions1.size() + regions2.size());

    // Add segments from file1 (regions1)
    for (int region_idx : unique_regions1) {
        unsigned int start_idx = regions1.regionStartIndex[region_idx];
        unsigned int end_idx = (region_idx == regions1.numRegions - 1) ? 
                              regions1.size() : regions1.regionStartIndex[region_idx + 1];
        
        for (unsigned int i = start_idx; i < end_idx; ++i) {
            all_segments.emplace_back(
                regions1.dx[i], regions1.dy[i], regions1.sx[i], regions1.sy[i],
                region_idx, i,
                regions1.interiorAbove[i], regions1.interiorBelow[i],
                0  // fileID for file1
            );
        }
    }

    // Add segments from file2 (regions2)
    for (int region_idx : unique_regions2) {
        unsigned int start_idx = regions2.regionStartIndex[region_idx];
        unsigned int end_idx = (region_idx == regions2.numRegions - 1) ? 
                              regions2.size() : regions2.regionStartIndex[region_idx + 1];
        
        for (unsigned int i = start_idx; i < end_idx; ++i) {
            all_segments.emplace_back(
                regions2.dx[i], regions2.dy[i], regions2.sx[i], regions2.sy[i],
                region_idx, i,
                regions2.interiorAbove[i], regions2.interiorBelow[i],
                1  // fileID for file2
            );
        }
    }

    cout << "Total segments to process: " << all_segments.size() << endl;

    // Prepare input for sortSweep - this is where we fix the issue
    thrust::host_vector<float> sx1(all_segments.size());
    thrust::host_vector<float> sy1(all_segments.size());
    thrust::host_vector<float> sx2(all_segments.size());
    thrust::host_vector<float> sy2(all_segments.size());
    thrust::host_vector<int> sia(all_segments.size());
    thrust::host_vector<int> sib(all_segments.size());
    thrust::host_vector<int> sredblue(all_segments.size());
    
    // CRITICAL FIX: Looking at the example code, sortSweep expects:
    // x1,y1: start point
    // x2,y2: end point 
    // From the example code: x2[i] = x1[i]+rand()%8000;
    // This shows x2,y2 are absolute end coordinates, not deltas
    for (size_t i = 0; i < all_segments.size(); ++i) {
        const Segment& seg = all_segments[i];
        sx1[i] = static_cast<float>(seg.dx);           // Start x
        sy1[i] = static_cast<float>(seg.dy);           // Start y
        sx2[i] = static_cast<float>(seg.sx);           // End x (not delta x)
        sy2[i] = static_cast<float>(seg.sy);           // End y (not delta y)
        sia[i] = seg.interiorAbove;
        sib[i] = seg.interiorBelow;
        sredblue[i] = seg.fileID;
    }
    
    cout << "\nSegment data interpretation:" << endl;
    cout << "Using correct format per example code - dx/dy are end coordinates, not deltas" << endl;
    
    // Double check with a few sample segments
    cout << "\nFirst 5 segments:" << endl;
    for (int i = 0; i < std::min(5, (int)all_segments.size()); i++) {
        cout << "Segment " << i << ": (" 
             << sx1[i] << "," << sy1[i] << ") -> ("
             << sx2[i] << "," << sy2[i] << "), color=" 
             << sredblue[i] << endl;
    }
    
    // Check segment distribution (red/blue)
    int red_count = 0, blue_count = 0;
    for (size_t i = 0; i < all_segments.size(); i++) {
        if (sredblue[i] == 0) red_count++;
        else blue_count++;
    }
    cout << "Red segments (from file1): " << red_count << endl;
    cout << "Blue segments (from file2): " << blue_count << endl;
    
    if (red_count == 0 || blue_count == 0) {
        cout << "ERROR: All segments are the same color! No intersections will be detected." << endl;
        return 1;
    }
    
    // Basic sanity checks on segments
    int invalid_count = 0;
    
    for (size_t i = 0; i < all_segments.size(); i++) {
        // Check for zero-length segments
        if (fabs(sx1[i] - sx2[i]) < 1e-6 && fabs(sy1[i] - sy2[i]) < 1e-6) {
            invalid_count++;
        }
    }
    
    if (invalid_count > 0) {
        cout << "Warning: Found " << invalid_count << " zero-length segments" << endl;
    }
    
    // Output vectors - allocate more space for potential intersections
    int output_size = min(static_cast<int>(all_segments.size() * 10), 10000000);
    thrust::host_vector<float> fx1(output_size);
    thrust::host_vector<float> fy1(output_size);
    thrust::host_vector<float> fx2(output_size);
    thrust::host_vector<float> fy2(output_size);
    thrust::host_vector<int> fia(output_size);
    thrust::host_vector<int> fib(output_size);
    thrust::host_vector<int> fredblue(output_size);

    auto sweep_start = std::chrono::high_resolution_clock::now();

    int num_intersects = sortSweep(
        sx1, sy1, sx2, sy2,
        sia, sib, sredblue,
        all_segments.size(),
        output_size,
        fx1, fy1, fx2, fy2,
        fia, fib, fredblue
    );

    auto sweep_end = std::chrono::high_resolution_clock::now();

    cout << "Number of intersections found: " << num_intersects << endl;
    
    // // Print some of the intersections for verification
    // if (num_intersects > 0) {
    //     cout << "\nFirst " << min(10, num_intersects) << " intersections:" << endl;
    //     for (int i = 0; i < min(10, num_intersects); i++) {
    //         cout << "Intersection " << i << ": (" << fx1[i] << "," << fy1[i] << ")" << endl;
    //     }
    // }

    auto end1 = std::chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    chrono::duration<double> elapsed1 = end1 - start;
    chrono::duration<double> elapsed_sweep = sweep_end - sweep_start;
    cout << "Total time for bounding box: " << elapsed.count() << " seconds" << endl;
    cout << "Total time for sortSweep: " << elapsed_sweep.count() << " seconds" << endl;
    cout << "Total time: " << elapsed1.count() << " seconds" << endl;

    return 0;
}