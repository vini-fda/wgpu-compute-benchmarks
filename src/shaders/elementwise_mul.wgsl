@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // the dimensions of x must match the dimensions of output as well as y
    let index = global_id.x;

    // output is the element-wise product of x and y
    output[index] = x[index] * y[index];
}