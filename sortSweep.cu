
#include "sortSweep.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <chrono>
#include <iostream>
#include <iomanip>
using namespace std;



void printFloatVec( const thrust::device_vector<float> & v, const int & size){
    
    for(int i = 0; i < size; i++ )
        cout << v[i] << ", ";
    cout << endl;
}
void printIntVec( const thrust::device_vector<int> & v, const int & size){
    
    for(int i = 0; i < size; i++ )
        cout << v[i] << ", ";
    cout << endl;
}


//typedef thrust::device_vector<float>::iterator floatIt;
//typedef thrust::device_vector<int>::iterator   intIt;
//typedef thrust::tuple<floatIt,floatIt,floatIt,floatIt,intIt,intIt> segIt ;
struct segCompare{
    __host__ __device__
    bool operator()(  const thrust::tuple<float, float, float, float, int, int, int> & s1,
                        const thrust::tuple<float, float, float, float, int, int, int> & s2)   {
        if( thrust::get<0>(s1) < thrust::get<0>(s2) 
                || (thrust::get<0>(s1) == thrust::get<0>(s2) && thrust::get<1>(s1) < thrust::get<1>(s2)) )
            return true;
        return false;
    }
};

struct splitSegCompare {
    __host__ __device__
    bool operator()(const thrust::tuple<float, float, int, int> & s1,
                    const thrust::tuple<float, float, int, int> & s2) {
        if(     thrust::get<0>(s1) < thrust::get<0>(s2) 
                || (thrust::get<0>(s1) == thrust::get<0>(s2) && thrust::get<1>(s1) < thrust::get<1>(s2))                           || (thrust::get<0>(s1) == thrust::get<0>(s2) && thrust::get<1>(s1) == thrust::get<1>(s2)
                    && thrust::get<3>(s1) < thrust::get<3>(s2) )
          )
            return true;
        return false;
    }
};
struct xorElement{
    __host__ __device__
    int operator()(const int& arg ){
        return (arg+1)%2;
    }
};

struct indexedPointCompare{
    __host__ __device__
    bool operator()( const thrust::tuple<int, float, float, int> &p1,
                        const thrust::tuple<int, float, float, int> &p2) {
        if( thrust::get<0>(p1) < thrust::get<0>(p2) 
                || (thrust::get<0>(p1) == thrust::get<0>(p2) && thrust::get<1>(p1) < thrust::get<1>(p2))
                || thrust::get<0>(p1) == thrust::get<0>(p2) && thrust::get<1>(p1) == thrust::get<1>(p2)
                        && thrust::get<2>(p1) < thrust::get<2>(p2))
            return true;
        return false;
    }
};

struct intersectorFunc {
    float x1,y1,x2,y2, *ox, *oy;
    const int id, rb;
    int * rayShootVals, *oid, *ocounts, *oTOS;
    __host__ __device__ intersectorFunc( float _x1, float _y1, float _x2, float _y2,
            const int _id, const int _rb, int *_rayShootVals, 
            int* _oid, float * _ox, float * _oy, int* _ocounts, int* _oTOS  ):
        x1(_x1), y1(_y1), x2(_x2), y2(_y2), id(_id), rb(_rb), rayShootVals(_rayShootVals),
        oid(_oid), ox(_ox), oy(_oy), ocounts(_ocounts), oTOS(_oTOS){}
    
    __host__ __device__
        float leftTurn( const float px0, const float py0, const float px1, const float py1, const float px2, const float py2) const {
            return ( (px0-px1)*(py2-py1)) - ( (py0-py1)*(px2-px1) );
        }

    __device__
    void operator()( thrust::tuple< float, float,float,float, int, int> t) {
        float rx1 = thrust::get<0>(t);
        float ry1 = thrust::get<1>(t);
        float rx2 = thrust::get<2>(t);
        float ry2 = thrust::get<3>(t);
        int rID = thrust::get<4>(t);
        int rRB = thrust::get<5>(t);
        //bool intersects = 0;
        float a, b, c, d, ixval, iyval, pointCalc;
        int spot;
        //if(rb == rRB) return; // only do the tests for segs from different regions.
        // turn tests
        a = leftTurn( x1,y1,x2,y2,rx1,ry1 );
        b = leftTurn( x1,y1,x2,y2,rx2,ry2 );
        c = leftTurn( rx1,ry1,rx2,ry2,x1,y1 );
        d = leftTurn( rx1,ry1,rx2,ry2,x2,y2 );
        // intersection test
        //intersects |= (a!=b) && (a*b<=0) && (c*d <=0 );
        // colinear intersection tests
        if( !a  && ((x1< rx1 && x2 >rx1) || (x1 == rx1 && y1 < ry1 && y2 > ry1))) { 
            // rseg has left endpoint on interior of lseg, so report rx1, ry1 breaks lseg
            // report the point
            spot = atomicAdd( oTOS, 1 );
            oid[spot] = id;
            ox[spot] = rx1;
            oy[spot] = ry1;
            if( b < 0) ocounts[spot]=2;         //rhs is above lseg
            else if (b > 0 ) ocounts[spot]=1;   // rhs is below lseg
            else ocounts[spot] = 0;             // rhs is collinear with lseg
        } else if( a && !b && ((x1 <rx2 && x2 > rx2) || (x1 == rx2 && y1 < ry2 && y2 > ry2))) {
                // cannot get here if segs are colinear, that is handled in the 1st IF 
                // rsegs right end point is on interior of lseg.  
                spot = atomicAdd( oTOS, 1 );
                if( a < 0) ocounts[spot] = 2;   // rhs is above lhs
                else ocounts[spot] = 1;         // rhs is below lseg
                oid[spot] = id;
                ox[spot] = rx2;
                oy[spot] = ry2;
        } else if( a*b < 0 && c*d < 0 ){ // interior intersection, report break point to both segs
            pointCalc = (((x1-rx1)*(ry1-ry2))-((y1-ry1)*(rx1-rx2)))/(((x1-x2)*(ry1-ry2))-((y1-y2)*(rx1-rx2)));
            ixval = x1 + (pointCalc*(x2-x1));
            iyval = y1 + (pointCalc*(y2-y1));
            // report the point
            spot = atomicAdd( oTOS, 2);
            oid[spot+1] = id;
            ox[spot+1] = ixval;
            oy[spot+1] = iyval;
            ocounts[spot+1]= 3;
            oid[spot] = rID;
            ox[spot] = ixval;
            oy[spot] = iyval;
            ocounts[spot] = 3;
        }
        
        // ray shoot test
        // 3 cases,  p1 is below,  p1 is on and p2 is below, p1 and p2 are on (colinear)
        if(  (x1 <= rx1) && (x2 > rx1) && (x1 != x2) ) { // spans end point and not vertical
            if( !a && !b ) atomicOr(rayShootVals+rID, 2);// colinear
            else if( a>=0 || (a==0 && b>0))  atomicXor( rayShootVals+rID, 1); //other 2 cases
        }        
        
        //printf("we are in: %d, %d, %f %f %f %f, %f %f %f %f spot = %d\n", id, rID, x1, y1, x2, y2, rx1, ry1, rx2, ry2, spot);
         
    }
};

struct intersectLaunchFunc {
    float *x1,*y1,*x2,*y2, *ox, *oy;
    int* IDs, *redBlue, *rayShootVals,*oTOS, *oid, *ocounts;
    const int size;
    intersectLaunchFunc( float* _x1, float* _y1, float* _x2, float* _y2,
                            int* _IDs, int* _rayShootVals, int* _redBlue, const int _size, 
                            int* _oid, float * _ox, float * _oy, int* _ocounts, int* _oTOS ) :
        x1(_x1), y1(_y1), x2(_x2), y2(_y2),  IDs(_IDs), 
        rayShootVals(_rayShootVals), redBlue( _redBlue ), size( _size),
        oid(_oid), ox(_ox), oy(_oy), ocounts(_ocounts), oTOS(_oTOS){}
     __device__
    void operator()( const thrust::tuple<int, int> & t  ) {
        int id = thrust::get<1>(t);
        int count = thrust::get<0>(t)-id;
        if( count > 1 ) 
        { 
            //cast to device pointers!
            thrust::device_ptr<float> dx1=thrust::device_pointer_cast(x1)+id;
            thrust::device_ptr<float> dy1=thrust::device_pointer_cast(y1)+id;
            thrust::device_ptr<float> dx2=thrust::device_pointer_cast(x2)+id;
            thrust::device_ptr<float> dy2=thrust::device_pointer_cast(y2)+id;
            thrust::device_ptr<int>   dID=thrust::device_pointer_cast(IDs)+id;
            thrust::device_ptr<int>   drb=thrust::device_pointer_cast(redBlue)+id;
            // now we launch a thruster for each of these to do the work 
            thrust::for_each( thrust::device,
                    thrust::make_zip_iterator(thrust::make_tuple(dx1+1, dy1+1, dx2+1, dy2+1, dID+1, drb+1) ),
                    thrust::make_zip_iterator(thrust::make_tuple(dx1+count, dy1+count, dx2+count, dy2+count, dID+count, drb+count) ),
                    intersectorFunc(*dx1, *dy1, *dx2, *dy2, *dID, *drb, rayShootVals, oid, ox,oy, ocounts, oTOS) );
        }
    }
};


int sortSweep( thrust::host_vector<float> &x1, thrust::host_vector<float> &y1,
                thrust::host_vector<float> &x2, thrust::host_vector<float> &y2,
                thrust::host_vector<int> &ia, thrust::host_vector<int> &ib,
                thrust::host_vector<int> &redblue, const int size, const int OUTPUT_SIZE,
                thrust::host_vector<float> &fx1, thrust::host_vector<float> &fy1,
                thrust::host_vector<float> &fx2, thrust::host_vector<float> &fy2,
                thrust::host_vector<int>   &fia, thrust::host_vector<int>   &fib,
                thrust::host_vector<int>   &fredblue )
{
    int numWake = 4096000;
    wakeGPUup1(numWake);

    numWake = 40960;
    wakeGPUup1(numWake);

    auto time_start = std::chrono::high_resolution_clock::now();

    thrust::device_vector<float> dx1(x1);
    thrust::device_vector<float> dy1(y1);
    thrust::device_vector<float> dx2(x2);
    thrust::device_vector<float> dy2(y2);
    thrust::device_vector<int> dia(ia);
    thrust::device_vector<int> dib(ib);
    thrust::device_vector<int> dRayShootVals(size,0); 
    thrust::device_vector<int> oid(OUTPUT_SIZE);
    thrust::device_vector<float> ox(OUTPUT_SIZE);
    thrust::device_vector<float> oy(OUTPUT_SIZE);
    thrust::device_vector<int> ocounts(OUTPUT_SIZE);
    thrust::device_vector<int> oTOS(1,0);
    thrust::device_vector<int> dRedBlue( redblue );

    auto time_transfer_to_finished =  std::chrono::high_resolution_clock::now();
    thrust::sort(thrust::device, 
            thrust::make_zip_iterator( thrust::make_tuple(dx1.begin(), dy1.begin(),
                                dx2.begin(), dy2.begin(), dia.begin(),
                                dib.begin(), dRedBlue.begin()) ),
            thrust::make_zip_iterator( thrust::make_tuple(dx1.end(), dy1.end(),
                                dx2.end(), dy2.end(), dia.end(),
                                dib.end(), dRedBlue.end() )),
            segCompare() );

    // next we need double it up!
    thrust::device_vector<float> dsx(size*2);
    thrust::device_vector<float> dsy(size*2);
    thrust::device_vector<int> dsid(size*2);
    thrust::device_vector<int> dslr(size*2);
    // copy the left end points
    thrust::copy( thrust::make_zip_iterator( thrust::make_tuple(dx1.begin(), dy1.begin()) ),
                    thrust::make_zip_iterator( thrust::make_tuple(dx1.end(), dy1.end())),
                    thrust::make_zip_iterator( thrust::make_tuple(dsx.begin(), dsy.begin())) );
    // copy the right end points
    thrust::copy( thrust::make_zip_iterator( thrust::make_tuple(dx2.begin(), dy2.begin()) ),
                    thrust::make_zip_iterator( thrust::make_tuple(dx2.end(), dy2.end())),
                    thrust::make_zip_iterator( thrust::make_tuple(dsx.begin()+size, dsy.begin()+size)) );
    // put the L/R point values.  Left end poitn gets a 1, Right end points get a 0
    thrust::fill_n( dslr.begin(), size, 1);
    thrust::fill_n( dslr.begin()+size, size, 0);
    // set the segment ID array
    // right now,  the 1st half of the array has all left end points,  the second
    // half has all right end points.  They are in segment ID order, 
    // so just create the array with ID numbers assending.
    thrust::sequence( dsid.begin(), dsid.begin()+size);
    thrust::sequence( dsid.begin()+size, dsid.end());
    
    // we got it all. in there.  now sort everything!
    
    thrust::sort(thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple( dsx.begin(),dsy.begin(),dsid.begin(),dslr.begin() )),
            thrust::make_zip_iterator(thrust::make_tuple( dsx.end(),dsy.end(),dsid.end(),dslr.end() )),
            splitSegCompare() );

    // invert the LR array to make a stencil for scattering after we do the inclusive scan
    // the number on the Right end point will be the number of segments that seg needs to 
    // look at for intersections.

    thrust::device_vector<int> dstencil( dslr);
    thrust::transform( dstencil.begin(), dstencil.end(), dstencil.begin(), xorElement());
    /*
    printFloatVec( dsx, size*2);
    printFloatVec( dsy, size*2);
    printIntVec( dstencil, size*2);
    printIntVec( dsid, size*2);
    */

    // do the inclusive scan! 
    thrust::inclusive_scan(dslr.begin(), dslr.end(), dslr.begin());
    /*
    printFloatVec( dsx, size*2);
    printFloatVec( dsy, size*2);
    printIntVec( dslr, size*2);
    printIntVec( dsid, size*2);
    */
    // now we will do a scatter_if using dstencil as a stencil to gather the number associated 
    // with all the right end points
    // using ID as a map
    thrust::device_vector<int> dWorkList(size);
    thrust::scatter_if(dslr.begin(), dslr.end(), dsid.begin(), dstencil.begin(), dWorkList.begin());
    //printIntVec( dWorkList, size);
         
    thrust::device_vector<int> dWorkIDs(size);
    thrust::sequence( dWorkIDs.begin(), dWorkIDs.end());

    auto time_worklist_finished = std::chrono::high_resolution_clock::now();
    // // The following code computes, on average,  how many segments each
    // // segment will be tested against for intersection.
    // // for large numbers of segments,  the sum reduction will overflow the int
    // thrust::device_vector<int> dWorkAmounts(size);
    // thrust::minus<int> op;
    // thrust::transform(thrust::device, dWorkList.begin(), dWorkList.end(), dWorkIDs.begin(), dWorkAmounts.begin(), op);
    // int result = thrust::reduce(thrust::device, dWorkAmounts.begin(), dWorkAmounts.end());
    // cout << "avg intersection amounts: " << result/size <<endl;
    
    // do a for each where the functor will launch a thrust call for only the work each threads needs to do
    thrust::for_each( thrust::device, 
            thrust::make_zip_iterator( thrust::make_tuple(dWorkList.begin(), dWorkIDs.begin())),
             thrust::make_zip_iterator( thrust::make_tuple(dWorkList.end(), dWorkIDs.end())),
             intersectLaunchFunc( thrust::raw_pointer_cast(dx1.data()),
                thrust::raw_pointer_cast(dy1.data()),
                thrust::raw_pointer_cast(dx2.data()),
                thrust::raw_pointer_cast(dy2.data()),
                thrust::raw_pointer_cast(dWorkIDs.data()),
                thrust::raw_pointer_cast(dRayShootVals.data()),
                thrust::raw_pointer_cast(dRedBlue.data()),
                size,
                thrust::raw_pointer_cast(oid.data()),
                thrust::raw_pointer_cast(ox.data()),
                thrust::raw_pointer_cast(oy.data()),
                thrust::raw_pointer_cast(ocounts.data()),
                thrust::raw_pointer_cast(oTOS.data()) )
             );
   
    auto time_intersections_finished = std::chrono::high_resolution_clock::now();
    int hInters = oTOS[0];
    int fSize = size + hInters;
    // we now have all the information to construct the overlay of the input regions
    // we now need to break the segments accourding to the output points, and 
    // assign the topology based on the ray shoots and the ocounts.

    // sort the output points by id, x, y
    thrust::sort( thrust::device,
                thrust::make_zip_iterator(thrust::make_tuple(oid.begin(), ox.begin(), 
                                                            oy.begin(), ocounts.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(oid.begin()+hInters, ox.begin()+hInters, 
                                                            oy.begin()+hInters, ocounts.begin()+hInters)),
                indexedPointCompare() 
            );
    // final question.  can we build segs on the GPU.  would be nice!  I think we can
    // unique the oIDs
    // launch a foreach where we find the index of each id in the oIDs, this gives a start point
    // prefix sum the start points to get the number of output slots for the broken segs
    // launch a foreach with the unique oids.  each one breaks up its associated seg.
    // finally, remove the broken segs from the input list.

    // or we do it on the cpu much more easy peasy and serial.  will do that for now.
    // copy the data back to the host
    thrust::host_vector<int> hRayShootVals(size); 
    thrust::host_vector<int>  hoid(hInters);
    thrust::host_vector<float> hox(hInters);
    thrust::host_vector<float> hoy(hInters);
    thrust::host_vector<int> hocounts(hInters);
    thrust::copy( dx1.begin(), dx1.end(), x1.begin() );
    thrust::copy( dy1.begin(), dy1.end(), y1.begin() );
    thrust::copy( dx2.begin(), dx2.end(), x2.begin() );
    thrust::copy( dy2.begin(), dy2.end(), y2.begin() );
    thrust::copy( dia.begin(), dia.end(), ia.begin() );
    thrust::copy( dib.begin(), dib.end(), ib.begin() );
    thrust::copy( dRedBlue.begin(), dRedBlue.end(), redblue.begin() );
  
    thrust::copy( dRayShootVals.begin(), dRayShootVals.end(), hRayShootVals.begin() );
    thrust::copy( oid.begin(), oid.begin()+hInters, hoid.begin() );
    thrust::copy( ox.begin(), ox.begin()+hInters, hox.begin() );
    thrust::copy( oy.begin(), oy.begin()+hInters, hoy.begin() );
    thrust::copy( ocounts.begin(), ocounts.begin()+hInters, hocounts.begin() );


    /*
    cerr<< " copied back vals:"<<endl;
    printFloatVec( x1, size);
    printFloatVec( y1, size);
    printFloatVec( x2, size);
    printFloatVec( y2, size);
    printIntVec( ia, size);
    printIntVec( ib, size);
    printIntVec( redblue, size);
    //printIntVec( id, size );
    printIntVec( hRayShootVals, size );
    cerr<< " intersection points: "<< endl;
    cerr<<"id:\t"; printIntVec( hoid, hInters );
    cerr<<"x:\t"; printFloatVec( hox, hInters );
    cerr<<"y:\t"; printFloatVec( hoy, hInters );
    cerr<<"oc:\t"; printIntVec( hocounts, hInters );
    */

    // allocate space in the final output vectors
    fx1.reserve(fSize);
    fy1.reserve(fSize);
    fx2.reserve(fSize);
    fy2.reserve(fSize);
    fia.reserve(fSize);
    fib.reserve(fSize);
    fredblue.reserve(fSize);


    auto time_result_allocate_and_transfer_finished = std::chrono::high_resolution_clock::now();

    // for each original seg we create all its broken segs
    int z = 0, oCurr=0;
    int curria, currib;
    for( int i =0 ; i < size; i++ ) {
        // start with left end point of an original seg
        fx1[z] = x1[i];
        fy1[z] = y1[i];
        fia[z] = ia[i];
        fib[z] = ib[i];
        fredblue[z] = redblue[i];
        //if( hRayShootVals[i] == 0 ) no change in ia or ib 
        if ( hRayShootVals[i] == 1 ) { // inside other region
            fia[z]|=2;
            fib[z]|=2;
        }
        else if( hRayShootVals[i] == 2 ) // on a seg, only below is inside other region 
            fib[z]|=2;
        else if( hRayShootVals[i] == 3 ) //odd number above and on a seg, only above is inside other region
            fia[z] |=2;
        curria = fia[z];
        currib = fib[z];
        //nextia = 0;
        //nextib = 0;
        while( oCurr < hInters && hoid[oCurr] == i ){ // while a break point exists for seg i
            int endPointCounts = hocounts[ oCurr] ;
            while( oCurr < hInters-1 && hoid[oCurr] == hoid[oCurr+1] 
                    && hox[oCurr] == hox[oCurr+1] && hoy[oCurr] == hoy[oCurr+1]) { // combine counts for identical points
                endPointCounts ^= hocounts[ oCurr+1];
                oCurr++;
            }
            // break the segment
            fx2[z] = hox[oCurr];
            fy2[z] = hoy[oCurr];
            
            // start a new segment
            z++;
            fx1[z] = hox[oCurr];
            fy1[z] = hoy[oCurr];
            fredblue[z] = fredblue[z-1];
            // update ia/ib based on previous ia/ib and the ocount
            if( endPointCounts & 1 ) currib ^=2; // switch ib, entering/leaving other region
            if( endPointCounts & 2 ) curria ^= 2;// switch ia,  enterin/leaving other region
            fia[z] = curria;
            fib[z] = currib;
            oCurr++;
        }
        // finish off the segment
        fx2[z] = x2[i];
        fy2[z] = y2[i];
        z++;
    }

    auto time_break_segs_finished = std::chrono::high_resolution_clock::now();
    
    // cerr<<"final vals:"<<endl;
    // printFloatVec( fx1, z);
    // printFloatVec( fy1, z);
    // printFloatVec( fx2, z);
    // printFloatVec( fy2, z);
    // printIntVec( fia, z);
    // printIntVec( fib, z);
    // printIntVec( fredblue, z);
    
    // that should do it. 

    cout<<std::setw(6);
    std::chrono::duration<double> dataTrans = time_transfer_to_finished-time_start;
    std::chrono::duration<double> worklistCompute = time_worklist_finished - time_transfer_to_finished;
    std::chrono::duration<double> intersectionCompute = time_intersections_finished - time_worklist_finished;
    std::chrono::duration<double> resultAllocateTransfer = time_result_allocate_and_transfer_finished - time_intersections_finished;
    std::chrono::duration<double> breakSegs = time_break_segs_finished - time_result_allocate_and_transfer_finished;
    std::chrono::duration<double> totalTime = time_break_segs_finished - time_start; 
    cout << "number of intersections: " << hInters << endl
         << "times in seconds:" << endl  
         << "total time:                         " << totalTime.count() << endl
         << "data transfer host->device:         " << dataTrans.count() << endl
         << "compute work list:                  " << worklistCompute.count() << endl
         << "compute intersections and topology: " << intersectionCompute.count() << endl
         << "allocate and transfer device->host: " << resultAllocateTransfer.count() << endl
         << "create final segs and labels:       " << breakSegs.count() << endl 
         << endl;


    return hInters;
}


