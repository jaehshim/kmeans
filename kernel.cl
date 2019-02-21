__kernel void kmeans (
	__global float * data,
	__global float * cent,
	__global int * part,
	int class_n
	) {
	int i = get_global_id(0);
	int j;
	float min_dist = 1.7976931348623157E+308;
	float dist;
	float x, y;

	for (j = 0; j < class_n; j++) {
		x = data[i*2] - cent[j*2];				
		y = data[i*2+1] - cent[j*2+1];

		dist = x*x + y*y;

		if (dist < min_dist) {
			part[i] = j;
			min_dist = dist;
		}				
	}
}
__kernel void kmeans_2 (
        __global float * centroids,
        __global int * count,
	__local int * l_count
        ) {
        int i = get_global_id(0);
	int l_i = get_local_id(1);
	
	l_count[l_i] = count[i];
	barrier(CLK_LOCAL_MEM_FENCE);

	centroids[i*2] /= l_count[i];
	centroids[i*2+1] /= l_count[i];	
}
