#![feature(adt_const_params)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(variant_count)]
#![feature(const_refs_to_static)]

mod buffer;
mod framebuffer;
mod meshset;
mod rawbuffer;
mod texture;

pub mod uniform;
pub use math::{Mat4, Quat, Vec3};
pub use uniform::*;

pub mod shader;

use std::mem::MaybeUninit;

pub use framebuffer::*;
pub use glam as math;
pub use meshset::*;
pub use rawbuffer::*;
pub use texture::*;

mod __sealed {
    pub trait Sealed {}
    pub trait True {}
    pub enum Eval<const EXPR: bool> {}
    impl True for Eval<true> {}
}

#[derive(Clone, Copy)]
pub struct QuadVAO {
    id: u32,
}
impl QuadVAO {
    pub fn new() -> Self {
        let id;
        unsafe {
            let mut vao = std::mem::MaybeUninit::<u32>::uninit();
            gl::GenVertexArrays(1, vao.as_mut_ptr());
            panic_if_error();
            id = vao.assume_init();
            gl::BindVertexArray(id);
            panic_if_error();
            let mut buf = MaybeUninit::uninit();
            gl::GenBuffers(1, buf.as_mut_ptr());
            panic_if_error();
            let buf = buf.assume_init();
            gl::BindBuffer(gl::ARRAY_BUFFER, buf);
            panic_if_error();
            let data = [-1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
            gl::BufferData(
                gl::ARRAY_BUFFER,
                std::mem::size_of_val(&data) as _,
                data.as_ptr() as _,
                gl::STATIC_DRAW,
            );
            panic_if_error();
            gl::EnableVertexAttribArray(0);
            panic_if_error();
            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 0, 0 as _);
            panic_if_error();
        }
        Self { id }
    }
    pub fn bind<'a, T>(&'a self, _shader: &shader::ActiveShader<'_, T>) -> BoundQuadVAO<'a> {
        unsafe {
            gl::BindVertexArray(self.id);
            panic_if_error();
        }
        BoundQuadVAO(std::marker::PhantomData)
    }
}

pub struct BoundQuadVAO<'a>(std::marker::PhantomData<&'a ()>);
impl<'a> BoundQuadVAO<'a> {
    pub fn render(&self) {
        unsafe {
            gl::DrawArrays(gl::TRIANGLE_FAN, 0, 4);
            panic_if_error();
        }
    }
}

pub trait Render {
    type RenderParams<'a>
    where
        Self: 'a;
    fn render<'a>(&'a self, params: Self::RenderParams<'a>);
}

// pub trait SetUniforms {
//     fn set_uniforms(&self,) {}
// }

pub trait RenderMesh {
    type MeshId;
    fn render_mesh(&self, mesh: Self::MeshId);
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct VertexAttributeMetadata(u8);
impl VertexAttributeMetadata {
    pub const fn new(count: u8, ty: Type) -> Self {
        assert!(
            count >= 1 && count <= 4,
            "Number of attributes has to be 1-4"
        );

        Self(count << 4 | ty as u8)
    }
    pub const fn attr_count(&self) -> u8 {
        (self.0 >> 4) as u8
    }
    pub const fn ty(&self) -> Type {
        let ty = self.0 & 0xF;
        assert!(ty < 9, "Assertion failed: invalid type data!");
        unsafe { std::mem::transmute(ty) }
    }
    pub const fn size(&self) -> u8 {
        self.attr_count() * self.ty().size()
    }
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum Type {
    I8 = 0,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
    F16,
}
impl Type {
    pub const fn size(&self) -> u8 {
        match self {
            Type::I8 | Type::U8 => 1,
            Type::I16 | Type::U16 | Type::F16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::F64 => 8,
        }
    }
}
impl Into<gl::types::GLenum> for Type {
    fn into(self) -> gl::types::GLenum {
        match self {
            Self::I8 => gl::BYTE,
            Self::U8 => gl::UNSIGNED_BYTE,
            Self::I16 => gl::SHORT,
            Self::U16 => gl::UNSIGNED_SHORT,
            Self::I32 => gl::INT,
            Self::U32 => gl::UNSIGNED_INT,
            Self::F32 => gl::FLOAT,
            Self::F64 => gl::DOUBLE,
            Self::F16 => gl::HALF_FLOAT,
        }
    }
}

pub use gl::load_with as init_gl;

#[track_caller]
fn panic_if_error() {
    unsafe {
        match gl::GetError() {
            gl::NO_ERROR => (),
            e => {
                let msg = match e {
                    gl::INVALID_ENUM => "Invalid Enum",
                    gl::INVALID_VALUE => "Invalid Value",
                    gl::INVALID_OPERATION => "Invalid Operation",
                    gl::INVALID_FRAMEBUFFER_OPERATION => "Invalid Framebuffer Operation",
                    gl::OUT_OF_MEMORY => "Out of memory",
                    gl::STACK_OVERFLOW => "Stack overflow",
                    gl::STACK_UNDERFLOW => "Stack underflow",
                    _ => "Undefined",
                };
                panic!("GL Error: {}", msg);
            }
        }
    }
}

pub use shader_def::shaderdef_from_file as single_shaderdef_from_file;

#[macro_export]
macro_rules! shaderdef_from_file {
    ($($ident:ident {$($field:ident : $expr:expr),*$(,)?})*) => {
        $($crate::single_shaderdef_from_file!($crate $ident {$($field:$expr),*});)*
    };
}

pub trait RenderCap {
    const GL_PARAM: u32;
}
pub enum FaceCulling {}
impl RenderCap for FaceCulling {
    const GL_PARAM: u32 = gl::CULL_FACE;
}
pub enum DepthTest {}
impl RenderCap for DepthTest {
    const GL_PARAM: u32 = gl::DEPTH_TEST;
}

pub struct RenderDevice {}
impl RenderDevice {
    pub fn new() -> Self {
        Self {}
    }
    pub fn enable<T: RenderCap>(&self) {
        unsafe { gl::Enable(T::GL_PARAM) }
    }

    pub fn disable<T: RenderCap>(&self) {
        unsafe { gl::Disable(T::GL_PARAM) }
    }
    pub fn set_depth_fn(&self, depth_fn: DepthFn) {
        unsafe { gl::DepthFunc(depth_fn as u32) }
    }
}
pub trait Bind {
    type BoundObject<'a>
    where
        Self: 'a;
    fn bind<'a>(&'a mut self) -> Self::BoundObject<'a>;
}

pub trait ShaderInputsLayout {
    const LAYOUT: &'static [shader::InputType];
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VertexAttributeType {
    Position3D,
    Position2D,
    Normal,
    Tangent,
    Bitangent,
    UVCoord,
    Weights,
    Joints,
}
impl VertexAttributeType {
    pub const fn offsets(attributes: &[Self]) -> [u8; std::mem::variant_count::<Self>()] {
        let mut offsets = [255; std::mem::variant_count::<Self>()];
        let mut i = 0;
        let mut next_offset = 0;
        while i < attributes.len() {
            let attr = attributes[i];
            offsets[attr as usize] = next_offset;
            next_offset += attr.size();
            i += 1;
        }
        offsets
    }
    pub const fn attr_count(&self) -> u8 {
        match self {
            VertexAttributeType::Position3D => 3,
            VertexAttributeType::Position2D => 2,
            VertexAttributeType::Normal => 3,
            VertexAttributeType::Tangent => 3,
            VertexAttributeType::Bitangent => 3,
            VertexAttributeType::UVCoord => 2,
            VertexAttributeType::Weights => 4,
            VertexAttributeType::Joints => 4,
        }
    }
    pub const fn ty(&self) -> Type {
        match self {
            VertexAttributeType::Position3D => Type::F32,
            VertexAttributeType::Position2D => Type::F32,
            VertexAttributeType::Normal => Type::F32,
            VertexAttributeType::Tangent => Type::F32,
            VertexAttributeType::Bitangent => Type::F32,
            VertexAttributeType::UVCoord => Type::F32,
            VertexAttributeType::Weights => Type::F32,
            VertexAttributeType::Joints => Type::U8,
        }
    }
    pub const fn size(&self) -> u8 {
        self.attr_count() * self.ty().size()
    }
    pub const fn metadata(&self) -> VertexAttributeMetadata {
        VertexAttributeMetadata::new(self.attr_count(), self.ty())
    }
    pub const fn array_of_metadata<const N: usize>(
        layout: &[VertexAttributeType; N],
    ) -> [VertexAttributeMetadata; N] {
        let mut i = 0;
        let mut attrs = [VertexAttributeMetadata::new(1, Type::I8); N];
        while i < layout.len() {
            attrs[i] = layout[i].metadata();
            i += 1;
        }
        attrs
    }
    pub const fn shader_type(&self) -> shader::InputType {
        use shader::InputType;
        match self {
            VertexAttributeType::Position3D => InputType::Vec3,
            VertexAttributeType::Position2D => InputType::Vec2,
            VertexAttributeType::Normal => InputType::Vec3,
            VertexAttributeType::Tangent => InputType::Vec3,
            VertexAttributeType::Bitangent => InputType::Vec3,
            VertexAttributeType::UVCoord => InputType::Vec2,
            VertexAttributeType::Weights => InputType::Vec4,
            VertexAttributeType::Joints => InputType::IVec4,
        }
    }
}

#[macro_export]
macro_rules! declare_layouts {
    (@count $attr:ident) => {
        1
    };
    (@count $attr:ident $($rest:ident)*) => {
        1+$crate::declare_layouts!(@count $($rest)*)
    };
    ($($name: ident = $($attr:ident),*;)*) => {
        $(
            const $name: $crate::AttribLayout = {
                let vertex_attributes = &[
                    $(
                        $crate::VertexAttributeType::$attr
                    ),*
                ];
                let metadata = &[$($crate::VertexAttributeType::$attr.metadata()),*];
                let shader_layout = &[$($crate::VertexAttributeType::$attr.shader_type()),*];

                let offsets = $crate::VertexAttributeType::offsets(vertex_attributes);

                $crate::AttribLayout { vertex_attributes, metadata, shader_layout, offsets }
            };
        )*
    };
}
pub struct AttribLayout {
    pub vertex_attributes: &'static [VertexAttributeType],
    pub metadata: &'static [VertexAttributeMetadata],
    pub shader_layout: &'static [shader::InputType],
    pub offsets: [u8; std::mem::variant_count::<VertexAttributeType>()],
}

pub struct StaticMeshSetParams<'a> {
    pub index: usize,
    pub camera_transform: &'a Mat4,
}

// impl<'a, 'c> crate::Render for BoundMeshSet<'a, 'c, StaticMeshShader, MeshInfo> {
//     type RenderParams<'b> = StaticMeshSetParams<'b> where Self: 'b;

//     fn render<'b>(&'b self, params: Self::RenderParams<'b>) {
//         use crate::SetUniform;

//         let mesh = &self.meshes[params.index];
//         self.bound_mesh_buffer
//             .set_uniform(StaticMeshShader::albedo_map, mesh.extra().albedo_map());
//         // self.active_shader.set(
//         //     StaticMeshShader::normal_map,
//         //     mesh.normal_map(),
//         // );
//         // self.active_shader.set(
//         //     StaticMeshShader::roughness_map,
//         //     mesh.roughness_map(),
//         // );
//         //
//         self.bound_mesh_buffer
//             .set_uniform(StaticMeshShader::mvp, params.camera_transform);
//         self.draw_mesh(params.index);
//     }
// }

pub struct SkinnedMeshSetParams<'a> {
    pub static_params: StaticMeshSetParams<'a>,
    pub skin_matrices: &'a [Mat4],
    pub view_mat: &'a Mat4,
}

pub struct MeshInfo {
    albedo_map: Texture2D,
    normal_map: Texture2D,
    roughness_map: Texture2D,
}
impl MeshInfo {
    pub fn new(albedo_map: Texture2D, roughness_map: Texture2D, normal_map: Texture2D) -> Self {
        Self {
            albedo_map,
            roughness_map,
            normal_map,
        }
    }
    pub fn albedo_map(&self) -> Texture2D {
        self.albedo_map
    }
    pub fn normal_map(&self) -> Texture2D {
        self.normal_map
    }
    pub fn roughness_map(&self) -> Texture2D {
        self.roughness_map
    }
}

// impl<'a, 'c> crate::Render for BoundMeshSet<'a, 'c, SkinnedMeshShader, MeshInfo> {
//     type RenderParams<'b> = SkinnedMeshSetParams<'b> where Self: 'b;

//     fn render<'b>(&'b self, params: Self::RenderParams<'b>) {
//         use crate::SetUniform;

//         let mesh = &self.meshes[params.static_params.index];
//         self.bound_mesh_buffer
//             .set_uniform(SkinnedMeshShader::albedo_map, mesh.extra().albedo_map());
//         // self.active_shader.set(
//         //     StaticMeshShader::normal_map,
//         //     mesh.normal_map(),
//         // );
//         // self.active_shader.set(
//         //     StaticMeshShader::roughness_map,
//         //     mesh.roughness_map(),
//         // );
//         //

//         self.bound_mesh_buffer
//             .set_uniform(SkinnedMeshShader::joints, params.skin_matrices);
//         self.bound_mesh_buffer.set_uniform(
//             SkinnedMeshShader::mvp,
//             params.static_params.camera_transform,
//         );
//         self.bound_mesh_buffer
//             .active_shader
//             .set(SkinnedMeshShader::view_mat, params.view_mat);
//         self.draw_mesh(params.static_params.index);
//     }
// }
unsafe fn draw(indices_ptr: *const std::os::raw::c_void, elements_count: i32) {
    gl::DrawElements(gl::TRIANGLES, elements_count, gl::UNSIGNED_INT, indices_ptr);

    panic_if_error();
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    translation: Vec3,
    rotation: math::Quat,
}

impl Transform {
    pub const IDENTITY: Self = Self::new(Vec3::ZERO, Quat::IDENTITY);
    pub const fn new(translation: Vec3, rotation: math::Quat) -> Self {
        Self {
            translation,
            rotation,
        }
    }
    pub fn inverse(&self) -> Self {
        Self::new(self.translation * -1.0, self.rotation.inverse())
    }
    pub fn set_translation(&mut self, translation: Vec3) {
        self.translation = translation;
    }
    pub fn set_rotation(&mut self, rotation: math::Quat) {
        self.rotation = rotation;
    }
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_quat(self.rotation) * Mat4::from_translation(self.translation)
    }
}

impl approx::AbsDiffEq for Transform {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.translation.abs_diff_eq(other.translation, epsilon)
            && self.rotation.abs_diff_eq(other.rotation, epsilon)
    }
}
