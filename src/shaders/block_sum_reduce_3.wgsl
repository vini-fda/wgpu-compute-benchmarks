@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// TITLE: Reduction #3 - Sequential addressing
// DESCRIPTION:
// Here, we replace the strided indexing in the inner loop
// with reversed loop and thread_id based indexing.

// The workgroup size is the number of threads that will be executed in parallel in a given workgroup.
// This is supposed to be substituted before compilation
const WORKGROUP_SIZE = 256u;
// shared memory/shared data
// this is a block of memory that is shared between all the threads in a workgroup
var<workgroup> sdata: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {

    let tid = local_id.x;
    let i = global_id.x;
    sdata[tid] = input[i];

    workgroupBarrier();

    for (var s = WORKGROUP_SIZE/2u; s > 0u; s >>= 1u) {
        if tid < s {
            sdata[tid] += sdata[tid + s];
        }
        workgroupBarrier();
    }

    if tid == 0u {
        output[group_id.x] = sdata[0];
    }
}
