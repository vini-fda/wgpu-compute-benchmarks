

use burn::backend::Wgpu;
use burn::backend::wgpu::kernel;
use burn::tensor::Tensor;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use wgpu::util::DeviceExt;
use wgpu::*;
use wgpu_compute_benchmarks::kernels::block_sum_reduce_1::BlockSumReduce1;
use wgpu_compute_benchmarks::kernels::block_sum_reduce_2::BlockSumReduce2;
use wgpu_compute_benchmarks::kernels::block_sum_reduce_3::BlockSumReduce3;
use wgpu_compute_benchmarks::kernels::block_sum_reduce_4::BlockSumReduce4;
use wgpu_compute_benchmarks::kernels::block_sum_reduce_5::BlockSumReduce5;
use wgpu_compute_benchmarks::kernels::block_sum_reduce_6::BlockSumReduce6;
use wgpu_compute_benchmarks::kernels::block_sum_reduce_7::BlockSumReduce7;
use wgpu_compute_benchmarks::kernels::gpu_executor::GPUExecutor;
use wgpu_compute_benchmarks::kernels::kernel1d::Kernel1D;
use wgpu_compute_benchmarks::kernels::sum_reduce::SumReduce;

use wgpu_compute_benchmarks::shader_utils::DEVICE_LIMITS;

// Type alias for the backend to use.
type Backend = Wgpu;

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

    // set device limits
    let limits = device.limits();
    let _ = DEVICE_LIMITS.set(limits);

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
    const N: usize = 1_000;
    let group_description = format!("SumReduce [executed {}x]", N);
    let mut group = c.benchmark_group(&group_description);

    for p in 20..=28 {
        let xlen = 1 << p;
        let x = vec![1.0f32; xlen];

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
        let kernel = BlockSumReduce1::new();
        let sum_reduce =
            SumReduce::from_kernel(&device, &x_buffer, &tmp_buffer, &output_buffer, &kernel);

        let f = || {
            // polls the function until it is ready
            pollster::block_on(async {
                execute_gpu_inner_1::<N>(&device, &queue, &sum_reduce);
                let mut result = [0.0f32];
                get_data(
                    &mut result,
                    &output_buffer,
                    &staging_buffer,
                    &device,
                    &queue,
                )
                .await;
                result[0]
            });
        };
        group.throughput(criterion::Throughput::Elements((N * xlen) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(xlen), &(), |b, _| b.iter(f));
    }
    group.finish();
}

fn execute_gpu_inner_1<const N: usize>(device: &Device, queue: &Queue, sum_reduce: &SumReduce) {
    for _ in 0..N {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

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
    let mut group = c.benchmark_group("SumReduce [executed once]");
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.sample_size(50);

    for p in 20..=28 {
        let xlen = 1 << p;
        let x = vec![1.0f32; xlen];

        let cpu_impl = || x.iter().sum::<f32>();

        group.bench_with_input(BenchmarkId::new("CPU implementation", xlen), &(), |b, _| {
            b.iter(cpu_impl)
        });

        // GPU implementation
        {
            // we put this in its own scope so that the buffers are dropped before we run the Burn implementation
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

            macro_rules! generate_kernel_benches {
                ($($kernel_name:ident),*) => {
                    $(
                        let kernel = $kernel_name::new();
                        let f = || {
                            pollster::block_on(execute_gpu_inner_2(
                                &device,
                                &queue,
                                &kernel,
                                &staging_buffer,
                                &x_buffer,
                                &tmp_buffer,
                                &output_buffer,
                            ));
                        };
            
                        let benchmark_id = format!("My own GPU implementation with Kernel {}", kernel.name());
            
                        group.bench_with_input(
                            BenchmarkId::new(&benchmark_id, xlen),
                            &(),
                            |b, _| {
                                b.iter(f);
                            },
                        );
                    )*
                };
            }
            
            generate_kernel_benches!(BlockSumReduce1, BlockSumReduce2, BlockSumReduce3, BlockSumReduce4, BlockSumReduce5, BlockSumReduce6, BlockSumReduce7);
        }

        // implementation using burn
        let xt = Tensor::<Backend, 1>::from_data(x.as_slice());
        let burn_impl = || xt.clone().sum();

        group.bench_with_input(
            BenchmarkId::new("Burn implementation", xlen),
            &(),
            |b, _| b.iter(burn_impl),
        );
    }
    group.finish();
}

async fn execute_gpu_inner_2<K: Kernel1D>(
    device: &Device,
    queue: &Queue,
    kernel: &K,
    staging_buffer: &Buffer,
    x_buffer: &Buffer,
    tmp_buffer: &Buffer,
    output_buffer: &Buffer,
) -> f32 {
    let sum_reduce = SumReduce::from_kernel(device, x_buffer, tmp_buffer, output_buffer, kernel);
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

criterion_group!(benches, benchmark_gpu_2);
criterion_main!(benches);
