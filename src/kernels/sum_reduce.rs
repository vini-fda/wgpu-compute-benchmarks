use super::kernel::Kernel;
use derive_macro::Kernel;
use wgpu::*;

use super::execution_step::ExecutionStep;

/// Represents a parallel sum-reduction operation on GPU buffers.
#[derive(Kernel)]
pub struct SumReduce {
    block_sum_reduce: ExecutionStep,
    sum_reduce_final: ExecutionStep,
}

impl SumReduce {
    pub fn new(
        device: &Device,
        x: &Buffer,
        tmp: &Buffer,
        output: &Buffer, // a single element buffer
        shader_source: &str,
    ) -> Self {
        let (block_sum_reduce, prev_stage_workgroups) = Self::first_stage(device, x, tmp, shader_source);
        let sum_reduce_final = Self::second_stage(device, tmp, output, prev_stage_workgroups);
        Self {
            block_sum_reduce,
            sum_reduce_final,
        }
    }

    fn first_stage(device: &Device, x: &Buffer, tmp: &Buffer, shader_source: &str) -> (ExecutionStep, u32) {
        let n = x.size() as u32 / std::mem::size_of::<f32>() as u32;
        let (workgroup_size, workgroups) = Self::workgroup_info(device, n);
        let shader = Self::load_shader_1(device, workgroup_size, shader_source);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Parallel sum-reduction pipeline (pass 1)"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bind group for parallel sum-reduction (pass 1)"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: tmp.as_entire_binding(),
                },
            ],
        });
        // the amount of workgroups in each dimension
        // this is generally passed as an argument to `dispatch_workgroups`
        let workgroups = (workgroups, 1, 1);
        (
            ExecutionStep::new(bind_group, pipeline, workgroups),
            workgroups.0,
        )
    }

    fn load_shader_1(device: &Device, workgroup_size: u32, shader_source: &str) -> ShaderModule {
        let new_expr = format!("const WORKGROUP_SIZE = {}u;", workgroup_size);
        let shader_source = shader_source.replace("const WORKGROUP_SIZE = 256u;", &new_expr);
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Parallel sum-reduction shader (pass 1)"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        shader
    }

    fn second_stage(
        device: &Device,
        tmp: &Buffer,
        output: &Buffer, // a single element buffer
        prev_stage_workgroups: u32,
    ) -> ExecutionStep {
        let shader = Self::load_shader_2(device, prev_stage_workgroups);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Parallel sum-reduction pipeline (pass 2)"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bind group for parallel sum-reduction (pass 2)"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: tmp.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        // the amount of workgroups in each dimension
        // this is generally passed as an argument to `dispatch_workgroups`
        let workgroups = (1, 1, 1);
        ExecutionStep::new(bind_group, pipeline, workgroups)
    }

    fn load_shader_2(device: &Device, prev_stage_workgroups: u32) -> ShaderModule {
        let shader_source = include_str!("../shaders/sum_reduce_final.wgsl");
        let new_expr = format!("const WORKGROUPS = {}u;", prev_stage_workgroups);
        let shader_source = shader_source.replace("const WORKGROUPS = 1u;", &new_expr);
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Parallel sum-reduction shader (pass 2)"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        shader
    }

    pub fn workgroup_info(device: &Device, n: u32) -> (u32, u32) {
        use crate::math_utils::{log2_ceil, next_closest_power_of_two};
        let workgroup_size = device.limits().max_compute_workgroup_size_x;
        let work_per_thread = 8 * next_closest_power_of_two(log2_ceil(n));
        let workgroups = n.div_ceil(work_per_thread * workgroup_size);
        // debug info
        #[cfg(debug_assertions)]
        {
            println!("n: {}", n);
            println!("workgroup_size: {}", workgroup_size);
            println!("work_per_thread: {}", work_per_thread);
            println!("workgroups: {}", workgroups);
        }

        (workgroup_size, workgroups)
    }
}

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;
    use wgpu::*;

    use crate::kernels::kernel::Kernel;
    use crate::shader_utils;

    use super::SumReduce;
    const ERR_DID_NOT_FIND_ADAPTER: &str = "Failed to find an appropriate adapter";

    async fn wgpu_init() -> (Device, Queue) {
        // Instantiates instance of WebGPU
        let instance = Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .expect(ERR_DID_NOT_FIND_ADAPTER);
        let best_limits = adapter.limits();

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: Features::empty(),
                    limits: best_limits,
                },
                None,
            )
            .await
            .unwrap();

        let info = adapter.get_info();
        // skip this on LavaPipe temporarily
        if info.vendor == 0x10005 {
            panic!("LavaPipe not supported")
        } else {
            (device, queue)
        }
    }

    async fn execute_gpu(x: &[f32], first_stage: &str) -> f32 {
        // Initialization
        let (device, queue) = wgpu_init().await;

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Buffer for reading results"),
            size: std::mem::size_of::<f32>() as BufferAddress,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let x_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Input buffer"),
            contents: bytemuck::cast_slice(x),
            usage: BufferUsages::STORAGE,
        });
        let tmp_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Temporary buffer"),
            size: x.len() as u64 * std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Output buffer"),
            size: std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        execute_gpu_inner(
            &device,
            &queue,
            &staging_buffer,
            &x_buffer,
            &tmp_buffer,
            &output_buffer,
            first_stage,
        )
        .await
    }

    async fn execute_gpu_inner(
        device: &Device,
        queue: &Queue,
        staging_buffer: &Buffer,
        x_buffer: &Buffer,
        tmp_buffer: &Buffer,
        output_buffer: &Buffer,
        first_stage: &str,
    ) -> f32 {
        let sum_reduce = SumReduce::new(device, x_buffer, tmp_buffer, output_buffer, first_stage);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        sum_reduce.add_to_pass(&mut cpass);
        drop(cpass);

        encoder.copy_buffer_to_buffer(
            output_buffer,
            0,
            staging_buffer,
            0,
            std::mem::size_of::<f32>() as BufferAddress,
        );

        queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..);
        // let (sender, receiver) = flume::bounded(1);
        // buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        // device.poll(wgpu::Maintain::Wait);
        // receiver.recv_async().await.unwrap().unwrap();
        // output.copy_from_slice(bytemuck::cast_slice(&buffer_slice.get_mapped_range()[..]));
        // staging_buffer.unmap();
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(Maintain::Wait);
        if let Ok(Ok(())) = receiver.recv_async().await {
            let result = staging_buffer.slice(..).get_mapped_range();
            let result: &[f32] = bytemuck::cast_slice(&result[..]);
            result[0]
        } else {
            panic!("Failed to read result from GPU")
        }
    }

    fn test_sum_reduce(shader_source: &str) {
        let x: Vec<f32> = vec![2.0; 1 << 20];
        let result = pollster::block_on(execute_gpu(&x, shader_source));
        assert_eq!(result, 2.0 * (1 << 20) as f32);
    }

    #[test]
    fn test_sum_reduce_1() {
        test_sum_reduce(include_str!("../shaders/block_sum_reduce_1.wgsl"));
    }

    #[test]
    fn test_sum_reduce_2() {
        test_sum_reduce(include_str!("../shaders/block_sum_reduce_2.wgsl"));
    }

    #[test]
    fn test_sum_reduce_3() {
        test_sum_reduce(include_str!("../shaders/block_sum_reduce_3.wgsl"));
    }

    #[test]
    fn test_sum_reduce_4() {
        test_sum_reduce(include_str!("../shaders/block_sum_reduce_4.wgsl"));
    }

    #[test]
    fn test_sum_reduce_5() {
        test_sum_reduce(include_str!("../shaders/block_sum_reduce_5.wgsl"));
    }

    #[test]
    fn test_sum_reduce_6() {
        let main_loop = shader_utils::gen_wgsl_main_unrolled_loop(1024, 32);
        let last_warp = shader_utils::gen_wgsl_last_warp(1024, 32);
        let shader6 = include_str!("../shaders/block_sum_reduce_6_template.wgsl");
        let shader6 = shader6.replace("//#main_loop", &main_loop);
        let shader6 = shader6.replace("//#last_warp", &last_warp);
        println!("{}", shader6);
        test_sum_reduce(&shader6);
    }

    #[test]
    fn test_sum_reduce_7() {
        let main_loop = shader_utils::gen_wgsl_main_unrolled_loop(1024, 32);
        let last_warp = shader_utils::gen_wgsl_last_warp(1024, 32);
        let shader7 = include_str!("../shaders/block_sum_reduce_7_template.wgsl");
        let shader7 = shader7.replace("//#main_loop", &main_loop);
        let shader7 = shader7.replace("//#last_warp", &last_warp);
        test_sum_reduce(&shader7);
    }
}
