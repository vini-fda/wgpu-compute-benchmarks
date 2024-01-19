use wgpu::{ShaderModuleDescriptor, ShaderSource};

pub trait Kernel1D {
    fn name(&self) -> &str;
    fn shader_source(&self) -> &str;
    fn shader_module(&self, device: &wgpu::Device) -> wgpu::ShaderModule {
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some(self.name()),
            source: ShaderSource::Wgsl(self.shader_source().into()),
        })
    }
    fn work_group_info(&self, n: u32) -> KernelExecutionInfo;
}

pub struct KernelExecutionInfo {
    pub workgroups: (u32, u32, u32),
    pub workgroup_size: (u32, u32, u32),
}