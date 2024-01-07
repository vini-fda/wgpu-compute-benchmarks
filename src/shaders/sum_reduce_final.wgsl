@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: f32;

// Number of workgroups that were executed in the previous pass
// (i.e. in the `block_sum_reduce` shader)
// This is supposed to be substituted before compilation
const WORKGROUPS = 1u;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // does the final pass of the sum reduction of the input array
    // and stores the final result in the output array
    // note that this is not executed in parallel, but only by a single thread
    let id = global_id.x;

    if (id == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < WORKGROUPS; i += 1u) {
            sum += input[i];
        }
        output = sum;
    }
}