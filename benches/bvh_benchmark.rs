use cloth_simulator_rust::mesh::simple::Mesh;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn bvh_mesh_build_benchmark(c: &mut Criterion) {
    let mut mesh = Mesh::new();
    mesh.read(&std::path::Path::new("models/monkey_subd_04.obj"))
        .unwrap();

    let epsilon = 0.01;
    c.bench_with_input(
        criterion::BenchmarkId::new("bvh_mesh_build", epsilon),
        &epsilon,
        |b, epsilon| {
            b.iter(|| {
                mesh.build_bvh(black_box(*epsilon));
            })
        },
    );
}

criterion_group!(benches, bvh_mesh_build_benchmark);
criterion_main!(benches);
