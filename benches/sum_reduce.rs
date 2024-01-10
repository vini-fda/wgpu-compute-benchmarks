use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wgpu::*;
use wgpu::util::DeviceExt;
use wgpu_compute_benchmarks::kernels::kernel::Kernel;
use wgpu_compute_benchmarks::kernels::sum_reduce::SumReduce;

const ERR_DID_NOT_FIND_ADAPTER: &str = "Failed to find an appropriate adapter";

async fn wgpu_init() -> (Device, Queue) {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect(ERR_DID_NOT_FIND_ADAPTER);
    let best_limits = adapter.limits();

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
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

fn benchmark_gpu_1(c: &mut Criterion) {
    // Initialization
    let (device, queue) = pollster::block_on(wgpu_init());

    let x = vec![1.0f32; 1 << 20];

    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Buffer for reading results"),
        size: std::mem::size_of::<f32>() as BufferAddress,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let x_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("Input buffer"),
        contents: bytemuck::cast_slice(&x),
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

    let sum_reduce = SumReduce::new(&device, &x_buffer, &tmp_buffer, &output_buffer);
    const N: usize = 10_000;

    let f = || {
        // polls the function until it is ready
        pollster::block_on(async {
            execute_gpu_inner_1::<N>(&device, &queue,  &sum_reduce);
            let mut result = [0.0f32];
            get_data(&mut result, &output_buffer, &staging_buffer, &device, &queue).await;
            result[0]
        });
    };
    let fn_description = format!("Parallel sum-reduce [executed {}x]", N);
    c.bench_function(&fn_description, |b| b.iter(f));
}

fn execute_gpu_inner_1<const N: usize>(
    device: &Device,
    queue: &Queue,
    sum_reduce: &SumReduce,
) {
    for _ in 0..N {
        let mut encoder =
        device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        sum_reduce.add_to_pass(&mut cpass);
        drop(cpass);
        queue.submit(Some(encoder.finish()));
    }
}

fn benchmark_gpu_2(c: &mut Criterion) {
    // Initialization
    let (device, queue) = pollster::block_on(wgpu_init());

    let x = vec![1.0f32; 1 << 20];

    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Buffer for reading results"),
        size: std::mem::size_of::<f32>() as BufferAddress,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let x_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("Input buffer"),
        contents: bytemuck::cast_slice(&x),
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

    let f = || {
        // polls the function until it is ready
        pollster::block_on(execute_gpu_inner_2(&device, &queue, &staging_buffer, &x_buffer, &tmp_buffer, &output_buffer));
    };

    c.bench_function("GPU implementation of parallel sum-reduce", |b| b.iter(f));
}

async fn execute_gpu_inner_2(
    device: &Device,
    queue: &Queue,
    staging_buffer: &Buffer,
    x_buffer: &Buffer,
    tmp_buffer: &Buffer,
    output_buffer: &Buffer,
) -> f32 {
    let sum_reduce = SumReduce::new(device, x_buffer, tmp_buffer, output_buffer);
    let mut encoder =
    device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    sum_reduce.add_to_pass(&mut cpass);
    drop(cpass);

    encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, std::mem::size_of::<f32>() as BufferAddress);
    
    queue.submit(Some(encoder.finish()));

    let mut result = [0.0f32];
    get_data(&mut result, output_buffer, staging_buffer, device, queue).await;
    result[0]
}

#[inline(always)]
async fn get_data<T: bytemuck::Pod>(
    output: &mut [T],
    storage_buffer: &wgpu::Buffer,
    staging_buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        command_encoder.copy_buffer_to_buffer(
        storage_buffer,
        0,
        staging_buffer,
        0,
        std::mem::size_of_val(output) as u64,
    );
    queue.submit(Some(command_encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    receiver.recv_async().await.unwrap().unwrap();
    output.copy_from_slice(bytemuck::cast_slice(&buffer_slice.get_mapped_range()[..]));
    staging_buffer.unmap();
}

criterion_group!(benches, benchmark_gpu_1, benchmark_gpu_2);
criterion_main!(benches);