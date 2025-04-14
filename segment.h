#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/sequence.h>
#include <vector>
#include <thrust/extrema.h>
#ifndef SEGMENT_H
#define SEGMENT_H

struct bbox{
    double lx,ly,ux,uy;
    bbox(): lx(0),ly(0),ux(0), uy(0)
    {   }
};

struct segment{
    double dx;
    double dy;
    double sx;
    double sy;
    int regionID;  // in the original region.
    int segID;     // store the original index of this segment
    int interiorAbove;
    int interiorBelow;
    void enforceOrdering(){
        if(sx < dx || (sx == dx && sy < dy )){
            double tempx = dx;
            double tempy = dy;
            dx = sx;
            dy = sy;
            sx = tempx;
            sy = tempy;
        }
    }
};

struct segments{
    // the left most end point is the ''dominating'' end point.
    // ALWAYS make sure the left most point is in the (dx,dy).
    //
    // There are lots of functi ons that depend on this assumption
    //
    // First region ID is 0, and IDs increment from there.

    thrust::host_vector<double> dx, dy, sx, sy;  // the segment end points
    thrust::host_vector<double> lx, ly, ux, uy;  // the bounding box lower left and upper right points
    thrust::host_vector<int> regionID, segID, interiorAbove, interiorBelow;
    thrust::host_vector<int> regionStartIndex; // each entry has the index of the 1st segment of the region with the id of index+1
    int numRegions;
    bbox globalBox;

    segments( ):numRegions(0)
    {   }


    void enforceOrdering(int index) {
        if( index >= dx.size() ){
            std::cerr<< "Index: " << index << "out of bounds for " << dx.size() << " segments"<<std::endl;
            exit(0);
        }
        if(sx[index] < dx[index] || (sx[index] == dx[index] && sy[index] < dy[index] )){
            double tempx = dx[index];
            double tempy = dy[index];
            dx[index] = sx[index];
            dy[index] = sy[index];
            sx[index] = tempx;
            sy[index] = tempy;
        }
    }
    // computer the regionStartIndex vector
    // element 0 will have the index of the first segmen for region 1
    // element 1 will have the index of the first segment for region 2
    // etc
    // note that we assume regions will start with an ID of 0 and go sequentially.
    void computeRegionStartIndex(){
        //std::cerr<< "numRegions: " << numRegions <<std::endl;
        thrust::host_vector<int> stencil( dx.size() );
        thrust::adjacent_difference( regionID.begin(), regionID.end(), stencil.begin(), thrust::minus<int>() );
        stencil[0] =1;
        //for( int i = 0; i < 500; i++ )
        //    std::cerr<< i<< "-" << stencil[i] << ", ";
        thrust::host_vector<int> sequence( dx.size() );
        thrust::sequence( sequence.begin(), sequence.end(), 0 );
        regionStartIndex.resize( numRegions );
        thrust::scatter_if(sequence.begin(), sequence.end(), regionID.begin(), stencil.begin(), regionStartIndex.begin() );
        //std::cerr<<"\n start indexes\n";
        //for( int i = 0; i < 500; i++ )
        //    std::cerr<< regionStartIndex[i] << ", ";
    }

    // get the bbox around the regions in the indicated region ID range
    void computeGlobalBBox( ){
        double tmp;
        globalBox.lx = * thrust::min_element( dx.begin(), dx.end());
        globalBox.ly = * thrust::min_element( dy.begin(), dy.end()); 
        tmp  = * thrust::min_element( sy.begin(), sy.end());
        if( tmp < globalBox.ly ) globalBox.ly = tmp;
        globalBox.ux = * thrust::max_element( sx.begin(), sx.end()); 
        globalBox.uy = * thrust::max_element( dy.begin(), dy.end());
        tmp  = * thrust::max_element( sy.begin(), sy.end());
        if( tmp > globalBox.uy ) globalBox.uy = tmp;
    } 
    
    // sometimes the map will be in antoher quadrant,  so just translate it.
    void shiftToQuadrant1( ){

        double xMax = globalBox.ux;
        double xMin = globalBox.lx;
        double yMax = globalBox.uy;
        double yMin = globalBox.ly;
        for( unsigned int i = 0 ; i < dx.size(); i++ ){
            if( xMin < 0 ) { 
                dx[i] = dx[i] + abs(xMin) + 1;
                sx[i] = sx[i] + abs(xMin) + 1;
            }
            if( yMin < 0 ){
                dy[i] = dy[i] + abs(yMin) + 1;
                sy[i] = sy[i] + abs(yMin) + 1;
            }
        }
        if( xMin < 0 ) {
            xMax = xMax + abs(xMin) + 1;
            xMin = xMin + abs(xMin) + 1;
        }
        if( yMin < 0 ){
            yMax = yMax + abs(yMin) + 1;
            yMin = yMin + abs(yMin) + 1;
        }
        globalBox.ux = xMax;
        globalBox.uy = yMax;
        globalBox.lx = xMin;
        globalBox.ly = yMin;
    }

    //compute the bboxes for all the individual regions in this map
    void computeBBoxes (){
        bbox x;
        double tmp;
        lx.resize( numRegions );
        ly.resize( numRegions );
        ux.resize( numRegions );
        uy.resize( numRegions );
        for( unsigned int i = 0; i < numRegions; i++ ){
            unsigned int startIndex = regionStartIndex[i];
            unsigned int endIndex;
            if( i == numRegions-1) endIndex = dx.size();
            else endIndex = regionStartIndex[ i+1 ];
            
            x.lx = * thrust::min_element( dx.begin()+startIndex, dx.begin()+endIndex );
            x.ly = * thrust::min_element( dy.begin()+startIndex, dy.begin()+endIndex );
            tmp  = * thrust::min_element( sy.begin()+startIndex, sy.begin()+endIndex );
            if( tmp < x.ly ) x.ly = tmp;
            x.ux = * thrust::max_element( sx.begin()+startIndex, sx.begin()+endIndex );
            x.uy = * thrust::max_element( dy.begin()+startIndex, dy.begin()+endIndex );
            tmp  = * thrust::max_element( sy.begin()+startIndex, sy.begin()+endIndex );
            if( tmp > x.uy ) x.uy = tmp;
            lx[i] = x.lx;
            ly[i] = x.ly;
            ux[i] = x.ux;
            uy[i] = x.uy;
        }

    }

    bool lessThan( const int L, const int R ) const{
        return (dx[L] < dx[R] || (dx[L] == dx[R] && dy[L] < dy[R])
                || (dx[L] == dx[R] && dy[L] == dy[R] && sx[L] < sx[R])
                || (dx[L] == dx[R] && dy[L] == dy[R] && sx[L] == sx[R] && sy[L] < sy[R]));
    }

    // note, equality testing does NOT include regionID or stripID
    inline bool equals( const int L, const int R ) const {
        return( dx[L] == dx[R] 
                && dy[L] == dy[R] 
                && sx[L] == sx[R] 
                && sy[L] == sy[R] );
    }

    inline bool colinear( const int L, const int R ) const {
        // segs are colinear if left hand turn test returns
        // 0 for both lht(this.p1, this.p2, rhs.p1) and
        // lht(this.p1, this.p2, rhs.p2)
        return 0 == ( ((dy[R] - dy[L]) * (sx[L] - dx[L])) - ((sy[L] - dy[L]) * (dx[R] - dx[L])) ) &&
            0 == ( ((sy[R] - dy[L]) * (sx[L] - dx[L])) - ((sy[L] - dy[L]) * (sx[R] - dx[L])) ) ;
    }
    unsigned int size(){
        return dx.size();
    }
    
    void append( const std::vector<segment> & S ){
        for( unsigned int i = 0; i < S.size(); i++ ){
            dx.push_back(S[i].dx);
            dy.push_back(S[i].dy);
            sx.push_back(S[i].sx);
            sy.push_back(S[i].sy);
            regionID.push_back(S[i].regionID);
            segID.push_back(S[i].segID);
            interiorAbove.push_back(S[i].interiorAbove);
            interiorBelow.push_back(S[i].interiorBelow);
        }
        if( S.size() > 0 ){
            if( numRegions != S[0].regionID ){
                std::cerr<< "appending region to segments struct and regionID is out of order\n"
                    << "adding ID " << S[0].regionID << " expecting ID" << numRegions<< std::endl;
                exit( 0 );
            } 
            numRegions++;
        }

    }
    void print( const int index) {
        std::cout << "RegionID: " << regionID[index] << std::endl;
        std::cout << "SegID: " << segID[index] << std::endl;
        std::cout << "Segment: (" << dx[index] << "," << dy[index] << ")-(" << sx[index] << "," << sy[index] << ")" << std::endl;
    }

};


/*struct segmentHasher
{
    std::size_t operator()(const segment& s) const
    {
        using std::size_t;
        using std::hash;
        size_t seed = 0;
        seed ^= std::hash<double>()(s.dx) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= std::hash<double>()(s.dy) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= std::hash<double>()(s.sx) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= std::hash<double>()(s.sy) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        return seed;
    }
};*/


#endif /* segment_h */
