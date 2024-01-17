@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// TITLE: Reduction #7 - Multiple adds per thread
// DESCRIPTION:
// This shader does a parallel sum reduction of the input array
// and stores the block results in the output array

const WORKGROUP_SIZE = 256u;
var<workgroup> sdata: array<f32, WORKGROUP_SIZE>;

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

    if WORKGROUP_SIZE >= 1024u {
        if tid < 512u {
            sdata[tid] += sdata[tid + 512u];
        }
        workgroupBarrier();
    }
    if WORKGROUP_SIZE >= 512u {
        if tid < 256u {
            sdata[tid] += sdata[tid + 256u];
        }
        workgroupBarrier();
    }
    if WORKGROUP_SIZE >= 256u {
        if tid < 128u {
            sdata[tid] += sdata[tid + 128u];
        }
        workgroupBarrier();
    }
    if WORKGROUP_SIZE >= 128u {
        if tid < 64u {
            sdata[tid] += sdata[tid + 64u];
        }
        workgroupBarrier();
    }

    if tid < 32u {
        if WORKGROUP_SIZE >= 64u {
          sdata[tid] += sdata[tid + 32u];
        }
        if WORKGROUP_SIZE >= 32u {
          sdata[tid] += sdata[tid + 16u];
        }
        if WORKGROUP_SIZE >= 16u {
            sdata[tid] += sdata[tid + 8u];
        }
        if WORKGROUP_SIZE >= 8u {
            sdata[tid] += sdata[tid + 4u];
        }
        if WORKGROUP_SIZE >= 4u {
            sdata[tid] += sdata[tid + 2u];
        }
        if WORKGROUP_SIZE >= 2u {
            sdata[tid] += sdata[tid + 1u];
        }
    }

    if tid == 0u {
        output[group_id.x] = sdata[0];
    }
}
