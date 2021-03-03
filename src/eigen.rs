use cpp::cpp;

cpp! {{
    #include <iostream>
}}

pub fn print_from_cpp(string: std::ffi::CString) {
    let string_ptr = string.as_ptr();
    unsafe {
        cpp!([string_ptr as "const char *"] {
        std::cout << "from_cpp: " << string_ptr << std::endl;
        })
    }
}
