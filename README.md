In this research, I have extended the Per Segment Plane Sweep Line Segment Intersection
algorithm to determine the intersection of the line segments using the parallelization
provided by the GPUs. As we know, spatial join operations are fundamental in determining
relationships between various geographical objects and line segments in large-scale spatial
datasets. It was important in a wide range of applications, including Geographical
Information Systems (GIS), computer graphics, computational geometry, and scientific
data visualization. All the problems associated with polygon overlay operations look
challenging, but the problem we are working i.e., line segment intersection, is a core
challenge due to the computational complexity involved, especially when dealing with
large datasets like geographical datasets.
The core innovation of this research lies in the introduction of a bounding box-based
filtering mechanism. Instead of directly processing all line segments in the dataset, our
approach first computes axis-aligned bounding boxes (AABBs) around each geographical
subregion. Once the bounding boxes are created, we assume that bounding boxes
representing the same geographical regions do not overlap with one another. It means
that we do not have to compare the line segments associated with the same geographical
regions. It is such a great improvement. Another important improvement is that all
ii
the bounding boxes of one Geographical Region R1 will not intersect with the bounding
boxes of Region Geographical R2. So, comparing the bounding boxes present in each
region, which are in the 100K range, is computationally efficient than comparing millions
of line segments. By performing an initial intersection test on the bounding boxes,
we can drastically reduce the number of line segments that need to be processed by
the more computationally expensive sweep line algorithm. This filtering mechanism is
particularly effective in scenarios where the spatial distribution of line segments is sparse,
as it eliminates a significant portion of redundant intersection tests.
