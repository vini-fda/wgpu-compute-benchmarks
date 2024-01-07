@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// DESCRIPTION:
// This shader does a parallel sum reduction of the input array
// and stores the block results in the output array

// The workgroup size is the number of threads that will be executed in parallel in a given workgroup.
// This is supposed to be substituted before compilation
const WORKGROUP_SIZE = 256u;
// shared memory/shared data
// this is a block of memory that is shared between all the threads in a workgroup
var<workgroup> sdata: array<f32, WORKGROUP_SIZE>;

// The value of global_invocation_id is equal to workgroup_id * workgroup_size + local_invocation_id. 
// therefore, global_id.x == local_id.x + group_id.x * WORKGROUP_SIZE

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {

    let n = arrayLength(&input); // the length of the input array (which must be the same as the output array)
    let grid_size = WORKGROUP_SIZE * 2u * num_groups.x;
    let tid = local_id.x;

    for (var i = local_id.x + 2u * group_id.x * WORKGROUP_SIZE; i < n; i += grid_size) {
        sdata[tid] += input[i] + input[i + WORKGROUP_SIZE];
    }

    workgroupBarrier();

    for (var s = WORKGROUP_SIZE / 2u; s > 32u; s >>= 1u) {
        if tid < s {
            sdata[tid] += sdata[tid + s];
        }

        workgroupBarrier();
    }

    if tid < 32u {
        sdata[tid] += sdata[tid + 32u];
        sdata[tid] += sdata[tid + 16u];
        sdata[tid] += sdata[tid + 8u];
        sdata[tid] += sdata[tid + 4u];
        sdata[tid] += sdata[tid + 2u];
        sdata[tid] += sdata[tid + 1u];
    }

    if tid == 0u {
        output[group_id.x] = sdata[0];
    }
}
