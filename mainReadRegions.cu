//#include "sortSweep.h"
#include <iostream>
//#include <sstream>
#include <thrust/host_vector.h>
//#include "file.h"
#include "segment.h"
//#include "box.h"
//#include "procCmd.h"
#include "readData.h"

using namespace std;

int main(int argc, char* argv[]){


    // get the command line args
    if( argc != 3 ){
        cerr << "Usage: " << argv[0] << "[fil1.dxf] [ file2.dxf] " <<endl;
        exit(0);
    }

    string file1( argv[1] ), file2(argv[2]);

    /* File input */
    segments regions1, regions2;
    readData(file1, regions1);
    readData(file2, regions2);
    if( regions1.numRegions == 0 || regions2.numRegions == 0){
        std::cout << "Nothing read from one of the files" << regions1.numRegions << " -- " << regions2.numRegions << std::endl;
        exit(0);
    }
    // At this point we have read the two input files and have the collections of regiosn stored.
    
    // compute the index in the segments struct so we know the start index of the first segment for each region.
    
    std::cerr<< "numregiosn 1 and 2 : " << regions1.numRegions << "   " << regions2.numRegions <<std::endl;
    regions1.computeRegionStartIndex();
    regions2.computeRegionStartIndex();
    //  regions.regionStartIndex maps a region ID to its starting position in the array
    
    // get global bboxes

    
    regions1.computeGlobalBBox( );
    regions2.computeGlobalBBox( );

    // shift to quadrant 1
    regions1.shiftToQuadrant1( );
    regions2.shiftToQuadrant1( );  

   // finally, need to create bboxes for each individual region 
    regions1.computeBBoxes();
    regions2.computeBBoxes();

    // now the bboxes are in the lx,ly, ux, uy vectors in the struct
   // the global box is in the struct
   // the segments are in the sx, sy, sx, sy in the struct, along with labels
   // each region has its own ID,  each struct starts with ID = 0
   //  the regionStartIndex maps the region ID (the index of the vector) to the index of the 1st segment
    // of that region.


    // now do the stuff!

}
    
