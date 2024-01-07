use super::kernel::Kernel;
use derive_macro::Kernel;
use wgpu::*;

use super::execution_step::ExecutionStep;

/// Represents an element-wise multiplication of two vectors in GPU memory.
#[derive(Kernel)]
pub struct ElementwiseMultiplication {
    step: ExecutionStep,
}

impl ElementwiseMultiplication {
    /// Creates a new kernel which is intended to perform `output = x .* y;`.
    ///
    /// As this operation only works in 1D vectors, the `workgroup_size` parameter corresponds to a single dimension (i.e. x)
    /// in the shader execution.
    pub fn new(
        device: &Device,
        x: &Buffer,
        y: &Buffer,
        output: &Buffer,
        workgroup_size: u32,
    ) -> Self {
        let work_size = x.size() as u32 / std::mem::size_of::<f32>() as u32;
        let shader_source = include_str!("../shaders/elementwise_mul.wgsl");
        let new_expr = format!("const WORKGROUP_SIZE = {}u;", workgroup_size);
        let shader_source = shader_source.replace("const WORKGROUP_SIZE = 256u;", &new_expr);
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Element-wise vector multiplication shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Element-wise vector multiplication pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bind group for element-wise vector multiplication"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: y.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        // the amount of workgroups in each dimension
        // this is generally passed as an argument to `dispatch_workgroups`
        let workgroups = (work_size.div_ceil(workgroup_size), 1, 1);
        Self {
            step: ExecutionStep::new(bind_group, pipeline, workgroups),
        }
    }
}

#[cfg(test)]
mod tests {
    use wgpu::{util::DeviceExt, Device, Queue};

    use crate::kernels::kernel::Kernel;

    use super::ElementwiseMultiplication;
    const ERR_DID_NOT_FIND_ADAPTER: &str = "Failed to find an appropriate adapter";

    async fn wgpu_init() -> (Device, Queue) {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect(ERR_DID_NOT_FIND_ADAPTER);

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
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

    async fn execute_gpu(vec_a: &[f32], vec_b: &[f32]) -> Vec<f32> {
        let (device, queue) = wgpu_init().await;

        execute_gpu_inner(&device, &queue, vec_a, vec_b).await
    }

    async fn execute_gpu_inner(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vec_a: &[f32],
        vec_b: &[f32],
    ) -> Vec<f32> {
        // Gets the size in bytes of the buffer.
        let slice_size = vec_a.len() * std::mem::size_of::<u32>();
        let size = slice_size as wgpu::BufferAddress;

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instantiates buffer with data (`vec_a`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a bind group and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy.
        let storage_buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer A"),
            contents: bytemuck::cast_slice(vec_a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let storage_buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer B"),
            contents: bytemuck::cast_slice(vec_b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Stores the result of the computation
        let storage_buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Storage Buffer C"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        const WORKGROUP_SIZE: u32 = 256;
        let vecmul = ElementwiseMultiplication::new(
            device,
            &storage_buffer_a,
            &storage_buffer_b,
            &storage_buffer_c,
            WORKGROUP_SIZE,
        );

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        vecmul.add_to_pass(&mut cpass);
        drop(cpass);

        // Adds a copy operation to the command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&storage_buffer_c, 0, &staging_buffer, 0, size);

        // Submits command encoder for processing
        queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        device.poll(wgpu::Maintain::Wait);

        // Awaits until `buffer_future` can be read from
        if let Some(Ok(())) = receiver.receive().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
                                    // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                    //   delete myPointer;
                                    //   myPointer = NULL;
                                    // It effectively frees the memory

            // Returns data from buffer
            result
        } else {
            panic!("failed to run dot product compute on gpu!")
        }
    }

    /// Asserts that the result is equal to the correct
    /// element-wise multiplication of `vec_a` and `vec_b`
    fn assert_correct_mul(vec_a: &[f32], vec_b: &[f32]) {
        let vec_c = vec_a
            .iter()
            .zip(vec_b.iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<f32>>();
        let result = pollster::block_on(execute_gpu(vec_a, vec_b));
        assert!(result == vec_c);
    }

    /// Asserts that the result is not equal to `vec_c`
    fn assert_notequal(vec_a: &[f32], vec_b: &[f32], vec_c: &[f32]) {
        let result = pollster::block_on(execute_gpu(vec_a, vec_b));
        assert!(result.iter().zip(vec_c.iter()).any(|(a, b)| a != b));
    }

    #[test]
    fn elementwise_multiplication() {
        // test 1:
        let mut vec_a = vec![2.0; 128 * 128];
        let mut vec_b = vec![3.0; 128 * 128];
        assert_correct_mul(&vec_a, &vec_b);
        // test 2: now with random numbers
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for i in 0..128 * 128 {
            vec_a[i] = rng.gen::<f32>();
            vec_b[i] = rng.gen::<f32>();
        }
        assert_correct_mul(&vec_a, &vec_b);
        // test 3: now with random numbers and a different size that is not a power of 2
        let vec_a = (0..77).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
        let vec_b = (0..77).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
        assert_correct_mul(&vec_a, &vec_b);
        //test 4: assert that the result is not equal to another unrelated vector
        let vec_a = vec![1.0; 128 * 128];
        let vec_b = vec![0.0; 128 * 128];
        let vec_c = vec![-1.0; 128 * 128];
        assert_notequal(&vec_a, &vec_b, &vec_c);
    }
}
