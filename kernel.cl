__kernel void kmeans (
	__global float * data,
	__global float * centroid,
	__global int * part,
	int class_n
	) {
	int i = get_global_id(0);
	int j;
	float min_dist = 1.7976931348623157E+308;
	float dist;
	float x, y;

	for (j = 0; j < class_n; j++) {
		x = data[i*2] - centroid[j*2];				
		y = data[i*2+1] - centroid[j*2+1];

		dist = x*x + y*y;

		if (dist < min_dist) {
			part[i] = j;
			min_dist = dist;
		}				
	}
}
__kernel void kmeans_2 (
        __global float * centroids,
        __global int * count
        ) {
        int i = get_global_id(0);
	
	centroids[i*2] /= count[i];
	centroids[i*2+1] /= count[i];	
}
__kernel void kmeans_3 (
        __global float * centroids,
        __global int * partitioned,
        __global int * count,
	__global float * data
        ) {
        int data_i = get_global_id(0);

	centroids[partitioned[data_i]*2] += data[data_i*2];
	centroids[partitioned[data_i]*2+1] += data[data_i*2+1];
	count[partitioned[data_i]]++;
}
