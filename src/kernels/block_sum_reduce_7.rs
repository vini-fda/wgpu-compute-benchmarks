use crate::{shader_utils::{DEVICE_LIMITS, gen_wgsl_main_unrolled_loop, gen_wgsl_last_warp}, math_utils::{log2_ceil, next_closest_power_of_two}};

use super::kernel1d::{Kernel1D, KernelExecutionInfo};

pub struct BlockSumReduce7 {
    shader_source: String,
    workgroup_size: u32,
}

impl BlockSumReduce7 {
    pub fn new() -> Self {
        let warp_size = 32;
        let workgroup_size = DEVICE_LIMITS.get().unwrap().max_compute_workgroup_size_x;
        // fill the source code template
        let main_loop = gen_wgsl_main_unrolled_loop(workgroup_size, warp_size);
        let last_warp = gen_wgsl_last_warp(workgroup_size, warp_size);
        let shader_source = include_str!("../shaders/block_sum_reduce_7_template.wgsl");
        let shader_source = shader_source.replace("//#main_loop", &main_loop);
        let shader_source = shader_source.replace("//#last_warp", &last_warp);
        let replace = format!("const WORKGROUP_SIZE = {}u;", workgroup_size);
        let shader_source = shader_source.replace("const WORKGROUP_SIZE = 256u;", &replace);
        Self {
            shader_source: shader_source.to_string(),
            workgroup_size,
        }
    }
}

impl Default for BlockSumReduce7 {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel1D for BlockSumReduce7 {
    fn name(&self) -> &str {
        "block_sum_reduce_7"
    }
    fn shader_source(&self) -> &str {
        &self.shader_source
    }
    fn work_group_info(&self, n: u32) -> KernelExecutionInfo {
        let work_per_thread = 8 * next_closest_power_of_two(log2_ceil(n));
        KernelExecutionInfo {
            workgroups: (n.div_ceil(work_per_thread * self.workgroup_size), 1, 1),
            workgroup_size: (self.workgroup_size, 1, 1),
        }
    }
}