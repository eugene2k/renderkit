use super::*;

#[repr(u32)]
pub enum ShaderType {
    VertexShader = gl::VERTEX_SHADER,
    FragmentShader = gl::FRAGMENT_SHADER,
}
pub struct ActiveShader<'a, T> {
    store: &'a T,
    pub(crate) layout: &'static [InputType],
}

impl<'a, T> ActiveShader<'a, T> {
    pub fn new(store: &'a T, layout: &'static [InputType]) -> Self {
        Self { store, layout }
    }
}

pub trait SetUniform<U, V> {
    fn set(&self, uniform: U, val: V);
}

impl<'a, S, U, V> SetUniform<Uniform<S, U>, V> for ActiveShader<'a, S>
where
    S: UniformStore<U>,
    V: Set<U>,
    U: UniformTy,
{
    fn set(&self, var: Uniform<S, U>, val: V) {
        V::set(self.store.get_store()[var.idx], val)
    }
}

impl<'a, Store, TextureType>
    SetUniform<SamplerUniform<Store, TextureType::SamplerType>, TextureType>
    for ActiveShader<'a, Store>
where
    Store: UniformStore<TextureType::SamplerType>,
    TextureType: Texture,
{
    fn set(&self, var: SamplerUniform<Store, TextureType::SamplerType>, val: TextureType) {
        let location = self.store.get_store()[var.uniform.idx].0;
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + var.texture_unit);
            panic_if_error();
            gl::BindTexture(
                TextureType::SamplerType::GL_TEXTURE_TYPE,
                val.get_handle().handle.into(),
            );
            panic_if_error();
            gl::Uniform1i(location, var.texture_unit as i32);
            panic_if_error();
        }
    }
}
#[derive(Debug)]
pub enum ShaderError {
    IoError(std::io::Error),
    CompileError(String),
}
#[derive(Debug, Clone, Copy)]
pub struct ShaderObject(u32);
impl ShaderObject {
    pub fn new<P>(path: &P, shader_type: ShaderType) -> Result<Self, ShaderError>
    where
        for<'a> &'a P: AsRef<std::path::Path>,
        P: AsRef<std::path::Path> + ?Sized,
    {
        fn create_shader(
            source: String,
            shader_type: ShaderType,
        ) -> Result<ShaderObject, ShaderError> {
            unsafe {
                let shader = gl::CreateShader(shader_type as _);

                gl::ShaderSource(
                    shader,
                    1,
                    [source.as_bytes().as_ptr()].as_ptr() as _,
                    &(source.len() as i32),
                );
                gl::CompileShader(shader);
                let mut status = MaybeUninit::uninit();
                gl::GetShaderiv(shader, gl::COMPILE_STATUS, status.as_mut_ptr());
                if status.assume_init() != gl::TRUE as i32 {
                    let mut length = MaybeUninit::uninit();
                    gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, length.as_mut_ptr());
                    let mut length = length.assume_init();
                    let mut log = Vec::<u8>::with_capacity(length as _);
                    log.set_len(length as _);
                    gl::GetShaderInfoLog(shader, length, &mut length, log.as_mut_ptr() as _);
                    gl::DeleteShader(shader);
                    let error_string = String::from_utf8_unchecked(log);
                    Err(ShaderError::CompileError(error_string))
                } else {
                    Ok(ShaderObject(shader))
                }
            }
        }
        let source = std::fs::read_to_string(path).map_err(ShaderError::IoError)?;
        create_shader(source, shader_type)
    }
}
#[derive(Clone, Copy)]
pub struct ShaderProgram(u32);
impl ShaderProgram {
    pub fn new(shaders: &[ShaderObject]) -> Result<Self, String> {
        unsafe {
            let program = gl::CreateProgram();
            for &ShaderObject(shader) in shaders {
                gl::AttachShader(program, shader);
            }
            gl::LinkProgram(program);
            let mut status = MaybeUninit::uninit();
            gl::GetProgramiv(program, gl::LINK_STATUS, status.as_mut_ptr());
            if status.assume_init() != gl::TRUE as i32 {
                let mut length = MaybeUninit::uninit();
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, length.as_mut_ptr());
                let mut length = length.assume_init();
                let mut log = Vec::with_capacity(length as _);
                log.set_len(length as _);
                gl::GetProgramInfoLog(program, length, &mut length, log.as_mut_ptr() as _);
                gl::DeleteProgram(program);
                Err(String::from_utf8_unchecked(log))
            } else {
                Ok(ShaderProgram(program))
            }
        }
    }
    pub fn activate(&self) {
        unsafe { gl::UseProgram(self.0) }
    }
    pub fn get_uniform_location(&self, var_name: &str) -> UniformLocation {
        let mut var = [0u8; 256];
        unsafe {
            std::ptr::copy_nonoverlapping(var_name.as_ptr(), var.as_mut_ptr(), var_name.len());
            std::ptr::write_bytes(var.as_mut_ptr().add(var_name.len()), 0, 1);
        };
        unsafe { UniformLocation(gl::GetUniformLocation(self.0, var.as_ptr() as _)) }
    }
}

#[derive(Debug, PartialEq)]
pub enum InputType {
    Bool,
    Int,
    UInt,
    Float,
    Double,

    BVec2,
    BVec3,
    BVec4,

    IVec2,
    IVec3,
    IVec4,

    UVec2,
    UVec3,
    UVec4,

    Vec2,
    Vec3,
    Vec4,

    DVec2,
    DVec3,
    DVec4,

    Mat2,
    Mat2x3,
    Mat2x4,
    Mat3,
    Mat3x2,
    Mat3x4,
    Mat4,
    Mat4x2,
    Mat4x3,

    DMat2,
    DMat2x3,
    DMat2x4,
    DMat3,
    DMat3x2,
    DMat3x4,
    DMat4,
    DMat4x2,
    DMat4x3,
}
