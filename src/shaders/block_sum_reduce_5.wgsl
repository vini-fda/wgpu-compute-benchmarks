@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// TITLE: Reduction #5 - Unroll the last warp
// DESCRIPTION:
// We’re far from bandwidth bound, as we know reduction has a low arithmetic intensity.
// Therefore a likely bottleneck is instruction overhead (e.g. loop control, branching, etc.).
// We can reduce this by unrolling the last warp.

// This is supposed to be substituted before compilation
const WORKGROUP_SIZE = 256u;
var<workgroup> sdata: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>, @builtin(num_workgroups) num_groups: vec3<u32>) {

    let tid = local_id.x;
    let i = global_id.x + group_id.x * WORKGROUP_SIZE;
    sdata[tid] = input[i] + input[i + WORKGROUP_SIZE];

    workgroupBarrier();

    for (var s = WORKGROUP_SIZE/2u; s > 32u; s >>= 1u) {
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
