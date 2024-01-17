@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// TITLE: Reduction #2 - Interleaved addressing
// DESCRIPTION:
// Here, we replace the divergent branch in the inner loop
// with strided index and non-divergent branch.

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

    for (var s = 1; s < WORKGROUP_SIZE; s <<= 1u) {
        let index = 2u * s * tid;
        if index < WORKGROUP_SIZE {
            sdata[index] += sdata[index + s];
        }
        workgroupBarrier();
    }

    if tid == 0u {
        output[group_id.x] = sdata[0];
    }
}
