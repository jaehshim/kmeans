__kernel void calculate_distance_each_centroid(
        __constant float * g_data, 
        __global float * g_centroid, 
        __global int * g_partition, 
        __global float * g_distance,
        __local float * l_data, 
        __local float * l_centroid, 
        int data_n, int class_n ) 
{
	float min_dist = 1.7976931348623157E+308;
	float dist;
	float x, y;

    // copy data to l_data
	int i = get_global_id(0);
    int l_i = get_local_id(0);
    int g_i = get_group_id(0);
	int j, k;

    l_data[l_i*2] = (i < data_n) ? g_data[i*2] : 0;
    l_data[l_i*2+1] = (i < data_n) ? g_data[i*2+1] : 0;

	for (j = 0; j < class_n; j++) {
        l_centroid[j*2] = g_centroid[j*2];
        l_centroid[j*2+1] = g_centroid[j*2+1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate distance from each centroid
	for (j = 0; j < class_n; j++) {
		x = l_data[l_i*2] - l_centroid[j*2];				
		y = l_data[l_i*2+1] - l_centroid[j*2+1];

		dist = x*x + y*y;
        if (dist < min_dist) {
            g_partition[i] = j;
            min_dist = dist;
            g_distance[i] = min_dist;
        }
	}
}

__kernel void update_centroid(
        __global float * g_data
        )
{
}
