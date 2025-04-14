#include "readData.h"

//const std::string bluetty("\033[34m"); // tell tty to switch to blue
//const std::string deftty("\033[0m"); // tell tty to switch back to default color

#define PRINT(arg)  #arg ": " <<(arg)   // Print an expression's name then its
                                        // value, possibly
					// followed by a comma or std::endl.
					// Ex: std::cout << PRINTC(x) << PRINTN(y);
#define PRINTC(arg)  #arg << ": " << bluetty << (arg) << deftty << ", "
#define PRINTN(arg)  #arg << ": " << bluetty << (arg) << deftty << std::endl

/*
*  Reads in the DXF file and creates segments.
*/


void getPolyline(std::ifstream& readFile, int regionID, std::vector<segment> &polyline);
void getHatch(std::ifstream& readFile, int regionID, std::vector<segment> &polyline );
int getNumVertices(std::ifstream& readFile);
int getNumVertices(std::ifstream& readFile);
int getNumEdges(std::ifstream& readFile);
double getXValue(std::ifstream& readFile);
double getYValue(std::ifstream& readFile);
double getDoubleValue(std::ifstream& readFile, const int groupCode);
int getIntValue(std::ifstream& readFile, const int groupCode);
void setSegmentInteriorTags(std::vector<segment>& polyline);
int getPolylineOrientation(std::vector<segment>& polyline);


double colinearValue(const double p1x, const double p1y,
                                          const double p2x, const double p2y,
                                          const double p3x, const double p3y)
{
   return ((p3y - p1y) * (p2x - p1x)) - ((p2y - p1y) * (p3x - p1x));
}


void readData(std::string inputFile, segments & segmentsFromFile)
{
    int regionID = 0;
    //open the file
    std::ifstream readFile( inputFile );
    if( !readFile ){
        std::cerr << "error opening: " << inputFile << std::endl;
        exit(0);
    }
    segment currentSegment;
    std::string groupCodeLine;
    std::string dataLine;

    while (dataLine != "EOF")
    {
        std::vector<segment> polyline;
        getline(readFile, groupCodeLine);
        getline(readFile, dataLine);
        bool gotShape = false;
        if (dataLine == "LWPOLYLINE")
        {
            getPolyline(readFile, regionID, polyline);
            gotShape = true;
        }
        else if (dataLine == "HATCH")
        {
            getHatch(readFile, regionID, polyline);
            gotShape = true;
        }
        if( gotShape ){
            setSegmentInteriorTags(polyline);
            segmentsFromFile.append( polyline );
            regionID++;
        }
    }
    readFile.close();
    std::cout << "No. of Segments: " << segmentsFromFile.size() << "\n";
}

void getPolyline(std::ifstream& readFile, int regionID, std::vector<segment> &polyline)
{
   int numVertices;
   double x1, y1;
   double x2, y2;
   int segmentCounter = 0;
   segment currentSegment;
   
   numVertices = getNumVertices(readFile);
   x1 = getXValue(readFile);
   y1 = getYValue(readFile);
   x2 = getXValue(readFile);
   y2 = getYValue(readFile);
   for (int i = 2; i < numVertices; i++)
   {
      currentSegment.dx = x1;
      currentSegment.dy = y1;
      currentSegment.sx = x2;
      currentSegment.sy = y2;
      currentSegment.regionID = regionID;
      currentSegment.segID = ++segmentCounter;
      polyline.push_back(currentSegment);
      x1 = x2;
      y1 = y2;
      x2 = getXValue(readFile);
      y2 = getYValue(readFile);
   }
}

void getHatch(std::ifstream& readFile, int regionID, std::vector<segment> &polyline )
{
   int numEdges;
   double x1, y1;
   double x2, y2;
   int segmentCounter = 0;
   segment currentSegment;

   numEdges = getNumEdges(readFile);
   x1 = getXValue(readFile);
   y1 = getYValue(readFile);
   x2 = getXValue(readFile);
   y2 = getYValue(readFile);
   for (int i = 2; i <= numEdges; i++)
   {
      currentSegment.dx = x1;
      currentSegment.dy = y1;
      currentSegment.sx = x2;
      currentSegment.sy = y2;
      currentSegment.regionID = regionID;
      currentSegment.segID = ++segmentCounter;
      polyline.push_back(currentSegment);
      x1 = x2;
      y1 = y2;
      x2 = getXValue(readFile);
      y2 = getYValue(readFile);
   }
}

int getNumVertices(std::ifstream& readFile)
{
   const int NUM_VERTICES_GROUP_CODE = 90;

   return getIntValue(readFile, NUM_VERTICES_GROUP_CODE);
}

int getNumEdges(std::ifstream& readFile)
{
   const int NUM_EDGES_GROUP_CODE = 93;

   return getIntValue(readFile, NUM_EDGES_GROUP_CODE);
}

double getXValue(std::ifstream& readFile)
{
   const int X_VALUE_GROUP_CODE = 10;

   return  getDoubleValue(readFile, X_VALUE_GROUP_CODE);
}

double getYValue(std::ifstream& readFile)
{
   const int Y_VALUE_GROUP_CODE = 20;

   return  getDoubleValue(readFile, Y_VALUE_GROUP_CODE);
}

double getDoubleValue(std::ifstream& readFile, const int groupCode)
{
   std::string groupCodeLine = "0";
   std::string dataLine = "";

   while (stoi(groupCodeLine) != groupCode)
   {
      getline(readFile, groupCodeLine);
      getline(readFile, dataLine);
   }

   return stod(dataLine);
}

int getIntValue(std::ifstream& readFile, const int groupCode)
{
   std::string groupCodeLine = "0";
   std::string dataLine = "";

   while (stoi(groupCodeLine) != groupCode)
   {
      getline(readFile, groupCodeLine);
      getline(readFile, dataLine);
   }
   
   return stoi(dataLine);
}

void setSegmentInteriorTags(std::vector<segment>& polyline)
{
   int turnCounter;
   int rightSide;
   int leftSide;

   turnCounter = getPolylineOrientation(polyline);
   if (turnCounter > 0)  // Clockwise orientation
   {
      rightSide = 1;
      leftSide = 0;
   }
   else                // Counter-clockwise orientation
   {
      rightSide = 0;
      leftSide = 1;
   }

   for (int i = 0; i < polyline.size(); i++)
   {
      if (polyline[i].dx > polyline[i].sx)
      {
         polyline[i].interiorAbove = rightSide;
         polyline[i].interiorBelow = leftSide;
         polyline[i].enforceOrdering();
      }
      else if (polyline[i].dx < polyline[i].sx)
      {
         polyline[i].interiorAbove = leftSide;
         polyline[i].interiorBelow = rightSide;
      }
      else // dx == sx 
      {
         if (polyline[i].dy > polyline[i].sy)
         {
            polyline[i].interiorAbove = rightSide;
            polyline[i].interiorBelow = leftSide;
            polyline[i].enforceOrdering();
         }
         else
         {
            polyline[i].interiorAbove = leftSide;
            polyline[i].interiorBelow = rightSide;
         }
      } 
   }
   return;
}

int getPolylineOrientation(std::vector<segment>& polyline)
{
   int turnCounter = 0;
   
   for (int i = 0; i < polyline.size() - 1; i++)
   {
      //colinear value
      turnCounter = turnCounter +
                    colinearValue(polyline[i].dx, polyline[i].dy,
                                  polyline[i].sx, polyline[i].dy,
                                  polyline[i + 1].sx, polyline[i + 1].sy);
   }
   // Must count the final segment separately
   turnCounter = turnCounter + 
                 colinearValue(polyline[polyline.size() - 1].dx, polyline[polyline.size() - 1].dy,
                               polyline[polyline.size() - 1].sx, polyline[polyline.size() - 1].sy,
                               polyline[0].sx, polyline[0].sy);
   return turnCounter;
}
