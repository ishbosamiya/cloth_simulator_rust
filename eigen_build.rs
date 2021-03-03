extern crate cpp_build;

fn main() {
    let include_path = "deps/eigen";
    cpp_build::Config::new()
        .include(include_path)
        .build("src/eigen.rs");
}
