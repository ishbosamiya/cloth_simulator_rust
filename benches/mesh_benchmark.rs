use cloth_simulator_rust::mesh::simple::Mesh;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn mesh_generate_gl_mesh_benchmark(c: &mut Criterion) {
    let glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    let (mut window, _) = glfw
        .create_window(
            1280,
            720,
            "Mesh Generate GL Mesh Benchmark",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create glfw window");
    gl::load_with(|symbol| window.get_proc_address(symbol));

    let use_face_normal = false;
    let mut mesh = Mesh::new();
    mesh.read(&std::path::Path::new("models/monkey_subd_03.obj"))
        .unwrap();
    c.bench_with_input(
        criterion::BenchmarkId::new("mesh_generate_gl_mesh", use_face_normal),
        &use_face_normal,
        |b, use_face_normal| {
            b.iter(|| {
                mesh.generate_gl_mesh(black_box(*use_face_normal));
            })
        },
    );
}

criterion_group!(benches, mesh_generate_gl_mesh_benchmark,);
criterion_main!(benches);
