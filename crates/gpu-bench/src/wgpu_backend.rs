//! Portable GPU backend via `wgpu` — Metal on macOS, Vulkan on NVIDIA/AMD, DX12
//! on Windows.
//!
//! This measures **plain ALU throughput**: the GEMM kernel is a tiled matmul in
//! WGSL, not a tensor-core/matrix-core kernel (wgpu doesn't expose those). So
//! the numbers are honest but sub-peak — excellent for ranking devices and
//! sanity-checking, not a substitute for cuBLAS/MLX peak figures. Those land as
//! separate `Backend` implementations behind Cargo features.

use std::borrow::Cow;
use std::time::Instant;

use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::backend::{Backend, BandwidthResult, DeviceInfo, GemmResult, Precision};

/// 16×16 output tile per workgroup — matches `@workgroup_size(16, 16)` below.
const TILE: u32 = 16;

const GEMM_WGSL: &str = r#"
struct Dims { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TILE: u32 = 16u;
var<workgroup> tile_a: array<f32, 256>; // TILE*TILE
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
    let n = dims.n;
    let row = gid.y;
    let col = gid.x;
    var acc = 0.0;

    let num_tiles = (n + TILE - 1u) / TILE;
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        let b_row = t * TILE + lid.y;

        if (row < n && a_col < n) {
            tile_a[lid.y * TILE + lid.x] = a[row * n + a_col];
        } else {
            tile_a[lid.y * TILE + lid.x] = 0.0;
        }
        if (b_row < n && col < n) {
            tile_b[lid.y * TILE + lid.x] = b[b_row * n + col];
        } else {
            tile_b[lid.y * TILE + lid.x] = 0.0;
        }
        workgroupBarrier();

        for (var i = 0u; i < TILE; i = i + 1u) {
            acc = acc + tile_a[lid.y * TILE + i] * tile_b[i * TILE + lid.x];
        }
        workgroupBarrier();
    }

    if (row < n && col < n) {
        c[row * n + col] = acc;
    }
}
"#;

// Grid-stride triad: each thread walks the array in strides of the total thread
// count, so the workgroup count stays under the 65_535-per-dimension dispatch
// limit no matter how large the buffer is, while every element is touched once.
const TRIAD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups)       nwg: vec3<u32>) {
    let n = arrayLength(&c);
    let stride = nwg.x * 256u;
    var i = gid.x;
    while (i < n) {
        c[i] = a[i] + 2.0 * b[i];
        i = i + stride;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Dims {
    n: u32,
    _pad: [u32; 3],
}

/// A `wgpu` device ready to benchmark.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: DeviceInfo,
}

impl WgpuBackend {
    /// Select the highest-performance adapter and open a device on it.
    pub fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .context("no GPU adapter found")?;

        let info = adapter_to_info(&adapter);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu-bench"),
                required_features: wgpu::Features::empty(),
                // Request the adapter's real limits so large benchmark buffers fit.
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .context("failed to open GPU device")?;

        Ok(Self {
            device,
            queue,
            info,
        })
    }

    /// Enumerate every GPU adapter visible to `wgpu`.
    pub fn list() -> Vec<DeviceInfo> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        instance
            .enumerate_adapters(wgpu::Backends::all())
            .iter()
            .map(adapter_to_info)
            .collect()
    }

    fn storage_buffer(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Record `iterations` dispatches into one submission and wall-clock the GPU
    /// round trip. Batching amortizes submit overhead so the timing reflects
    /// kernel throughput, not driver bookkeeping.
    fn time_dispatches(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        groups: (u32, u32, u32),
        iterations: u32,
    ) -> f64 {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..iterations {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        let start = Instant::now();
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        start.elapsed().as_secs_f64()
    }

    /// Read `c[0]` back to the host — used to verify the GEMM produced the
    /// analytic result before trusting its throughput.
    fn read_first(&self, buffer: &wgpu::Buffer) -> anyhow::Result<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, 4);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let value = bytemuck::cast_slice::<u8, f32>(&data)[0];
        drop(data);
        staging.unmap();
        Ok(value)
    }
}

impl Backend for WgpuBackend {
    fn device_info(&self) -> DeviceInfo {
        self.info.clone()
    }

    fn gemm(
        &self,
        n: u32,
        precision: Precision,
        warmup: u32,
        iterations: u32,
    ) -> anyhow::Result<GemmResult> {
        // fp16 needs the SHADER_F16 feature and packed 16-bit storage; that is
        // the next increment. fp32 is the portable baseline.
        anyhow::ensure!(
            precision == Precision::F32,
            "the wgpu backend currently supports only fp32 GEMM; fp16 is the next step"
        );
        anyhow::ensure!(n > 0, "n must be > 0");

        let elems = (n as usize) * (n as usize);
        // A and B filled with 1.0 → every element of C must equal n. That makes
        // verification a single comparison and needs no reference matmul.
        let a = self.storage_buffer("a", &vec![1.0f32; elems]);
        let b = self.storage_buffer("b", &vec![1.0f32; elems]);
        let c = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c"),
            size: (elems * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::bytes_of(&Dims { n, _pad: [0; 3] }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gemm"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GEMM_WGSL)),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gemm"),
                layout: None,
                module: &module,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                binding(0, dims.as_entire_binding()),
                binding(1, a.as_entire_binding()),
                binding(2, b.as_entire_binding()),
                binding(3, c.as_entire_binding()),
            ],
        });

        let groups = (n.div_ceil(TILE), n.div_ceil(TILE), 1);

        if warmup > 0 {
            self.time_dispatches(&pipeline, &bind_group, groups, warmup);
        }
        let verified = (self.read_first(&c)? - n as f32).abs() < 0.5;
        let seconds = self.time_dispatches(&pipeline, &bind_group, groups, iterations);

        let flop = 2.0 * (n as f64).powi(3) * iterations as f64;
        let tflops = flop / seconds / 1e12;

        Ok(GemmResult {
            device: self.info.name.clone(),
            backend: self.info.backend.clone(),
            precision,
            n,
            iterations,
            seconds,
            tflops,
            verified,
        })
    }

    fn bandwidth(
        &self,
        elements: u64,
        warmup: u32,
        iterations: u32,
    ) -> anyhow::Result<BandwidthResult> {
        anyhow::ensure!(elements > 0, "elements must be > 0");
        let n = elements as usize;
        let a = self.storage_buffer("a", &vec![1.0f32; n]);
        let b = self.storage_buffer("b", &vec![2.0f32; n]);
        let c = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c"),
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("triad"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TRIAD_WGSL)),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("triad"),
                layout: None,
                module: &module,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("triad"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                binding(0, a.as_entire_binding()),
                binding(1, b.as_entire_binding()),
                binding(2, c.as_entire_binding()),
            ],
        });

        // Cap workgroups at the per-dimension dispatch limit; the grid-stride
        // kernel covers the rest of the array.
        let groups = ((elements as u32).div_ceil(256).min(65_535), 1, 1);
        if warmup > 0 {
            self.time_dispatches(&pipeline, &bind_group, groups, warmup);
        }
        let seconds = self.time_dispatches(&pipeline, &bind_group, groups, iterations);

        // Two reads (a, b) + one write (c) per element.
        let bytes_per_iter = elements * 3 * std::mem::size_of::<f32>() as u64;
        let gb_per_s = (bytes_per_iter as f64 * iterations as f64) / seconds / 1e9;

        Ok(BandwidthResult {
            device: self.info.name.clone(),
            backend: self.info.backend.clone(),
            bytes_per_iter,
            iterations,
            seconds,
            gb_per_s,
        })
    }
}

fn binding(slot: u32, resource: wgpu::BindingResource) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry {
        binding: slot,
        resource,
    }
}

fn adapter_to_info(adapter: &wgpu::Adapter) -> DeviceInfo {
    let info = adapter.get_info();
    DeviceInfo {
        name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
    }
}
