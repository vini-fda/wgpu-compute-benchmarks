use crate::shader_utils::DEVICE_LIMITS;

use super::kernel1d::{Kernel1D, KernelExecutionInfo};

pub struct BlockSumReduce5 {
    shader_source: String,
    workgroup_size: u32,
}

impl BlockSumReduce5 {
    pub fn new() -> Self {
        let workgroup_size = DEVICE_LIMITS.get().unwrap().max_compute_workgroup_size_x;
        let mut shader_source = include_str!("../shaders/block_sum_reduce_5.wgsl").to_string();
        let replace = format!("const WORKGROUP_SIZE = {}u;", workgroup_size);
        shader_source = shader_source.replace("const WORKGROUP_SIZE = 256u;", &replace);
        Self {
            shader_source,
            workgroup_size
        }
    }
}

impl Default for BlockSumReduce5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel1D for BlockSumReduce5 {
    fn name(&self) -> &str {
        "block_sum_reduce_5"
    }
    fn shader_source(&self) -> &str {
        &self.shader_source
    }
    fn work_group_info(&self, n: u32) -> KernelExecutionInfo {
        KernelExecutionInfo {
            workgroups: (n.div_ceil(2 * self.workgroup_size), 1, 1),
            workgroup_size: (self.workgroup_size, 1, 1),
        }
    }
}