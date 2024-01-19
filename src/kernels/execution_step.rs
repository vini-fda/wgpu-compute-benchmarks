use wgpu::{BindGroup, ComputePipeline, ComputePass};

use super::gpu_executor::GPUExecutor;

pub struct ExecutionStep {
    bind_group: BindGroup,
    pipeline: ComputePipeline,
    workgroups: (u32, u32, u32),
}

impl ExecutionStep {
    pub fn new(
        bind_group: BindGroup,
        pipeline: ComputePipeline,
        workgroups: (u32, u32, u32),
    ) -> Self {
        Self {
            bind_group,
            pipeline,
            workgroups,
        }
    }
}

impl GPUExecutor for ExecutionStep {
    fn add_to_pass<'a>(&'a self, pass: &mut ComputePass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.workgroups.0, self.workgroups.1, self.workgroups.2);
    }
}