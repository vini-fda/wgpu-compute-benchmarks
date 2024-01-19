use std::{sync::OnceLock};

/// Function to generate wgsl code like such, but with compile time eval of the constants:
/// ```wgsl
/// if WORKGROUP_SIZE >= 1024u {
///     if tid < 512u {
///         sdata[tid] += sdata[tid + 512u];
///     }
///     workgroupBarrier();
/// }
/// if WORKGROUP_SIZE >= 512u {
///     if tid < 256u {
///         sdata[tid] += sdata[tid + 256u];
///     }
///     workgroupBarrier();
/// }
/// if WORKGROUP_SIZE >= 256u {
///     if tid < 128u {
///         sdata[tid] += sdata[tid + 128u];
///     }
///     workgroupBarrier();
/// }
/// if WORKGROUP_SIZE >= 128u {
///     if tid < 64u {
///         sdata[tid] += sdata[tid + 64u];
///     }
///     workgroupBarrier();
/// }
/// ```
pub fn gen_wgsl_main_unrolled_loop(workgroup_size: u32, warp_size: u32) -> String {
    let mut code = String::new();
    let mut wg_size = workgroup_size;
    while wg_size >= 4 * warp_size {
        wg_size /= 2;
        code.push_str(&format!(
            "if tid < {}u {{\n    sdata[tid] += sdata[tid + {}u];\n}}\nworkgroupBarrier();\n",
            wg_size, wg_size
        ));
    }
    code
}

/// Function to generate wgsl code like such, but with compile time eval of the constants:
/// ```wgsl
///   if tid < 32u {
///   if WORKGROUP_SIZE >= 64u {
///     sdata[tid] += sdata[tid + 32u];
///   }
///   if WORKGROUP_SIZE >= 32u {
///     sdata[tid] += sdata[tid + 16u];
///   }
///   if WORKGROUP_SIZE >= 16u {
///       sdata[tid] += sdata[tid + 8u];
///   }
///   if WORKGROUP_SIZE >= 8u {
///       sdata[tid] += sdata[tid + 4u];
///   }
///   if WORKGROUP_SIZE >= 4u {
///       sdata[tid] += sdata[tid + 2u];
///   }
///   if WORKGROUP_SIZE >= 2u {
///       sdata[tid] += sdata[tid + 1u];
///   }
/// }
/// ```
pub fn gen_wgsl_last_warp(workgroup_size: u32, warp_size: u32) -> String {
    let mut code = format!("if tid < {}u {{\n", warp_size);
    let mut wg_size = workgroup_size;
    while wg_size > 2 * warp_size {
        wg_size /= 2;
    }
    while wg_size >= 2 {
        wg_size /= 2;
        code.push_str(&format!("sdata[tid] += sdata[tid + {}u];\n", wg_size));
    }
    code.push_str("}\n");
    code
}

pub static DEVICE_LIMITS: OnceLock<wgpu::Limits> = OnceLock::new();
