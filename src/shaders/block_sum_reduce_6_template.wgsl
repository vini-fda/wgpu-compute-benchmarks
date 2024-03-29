@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// TITLE: Reduction #6 - Completely Unrolled (Template)
// DESCRIPTION:
// We’re far from bandwidth bound, as we know reduction has a low arithmetic intensity.
// Therefore a likely bottleneck is instruction overhead (e.g. loop control, branching, etc.).
// We can reduce this by unrolling the last warp.

// This is a template file for the source code.
// The actual source code will be generated by Rust code in the `src` directory.

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

    //#main_loop

    //#last_warp

    if tid == 0u {
        output[group_id.x] = sdata[0];
    }
}
