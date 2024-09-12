use crate::{panic_if_error, shader::ActiveShader, shader::SetUniform, AttribLayout};
use std::num::{NonZero, NonZeroU32};

pub struct Mesh<T> {
    mesh_params: MeshParams,
    extra: T,
}
impl<T> Mesh<T> {
    pub fn new(mesh_params: MeshParams, extra: T) -> Self {
        Self { mesh_params, extra }
    }
    pub fn params(&self) -> &MeshParams {
        &self.mesh_params
    }
    pub fn extra(&self) -> &T {
        &self.extra
    }
}

pub struct MeshBuffer {
    vao: NonZeroU32,
    vbo: NonZeroU32,
    ibo: NonZeroU32,
    layout: &'static [super::shader::InputType],
}
impl MeshBuffer {
    pub fn new(vertices: &[u8], indices: &[u32], layout: &'static AttribLayout) -> Self {
        let vao;
        let vbo;
        let ibo;
        unsafe {
            let mut uninit_vao = std::mem::MaybeUninit::<NonZeroU32>::uninit();
            gl::GenVertexArrays(1, uninit_vao.as_mut_ptr() as _);
            panic_if_error();
            vao = uninit_vao.assume_init();
            gl::BindVertexArray(vao.get());
            panic_if_error();
            let mut buffers = std::mem::MaybeUninit::<[NonZero<u32>; 2]>::uninit();
            gl::GenBuffers(2, buffers.as_mut_ptr() as _);
            panic_if_error();
            [vbo, ibo] = buffers.assume_init();
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo.into());
            panic_if_error();
            gl::BufferData(
                gl::ARRAY_BUFFER,
                size_of_val(vertices) as isize,
                vertices.as_ptr().cast(),
                gl::STATIC_DRAW,
            );
            panic_if_error();
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ibo.into());
            panic_if_error();
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                size_of_val(indices) as isize,
                indices.as_ptr().cast(),
                gl::STATIC_DRAW,
            );
            panic_if_error();
        }
        let frame_size: u8 = layout.metadata.iter().map(|attr| attr.size()).sum();

        let mut offset = 0;
        for (i, attr) in layout.metadata.iter().enumerate() {
            unsafe {
                gl::EnableVertexAttribArray(i as u32);
                panic_if_error();
                match attr.ty() {
                    crate::Type::I8
                    | crate::Type::I16
                    | crate::Type::I32
                    | crate::Type::U8
                    | crate::Type::U16
                    | crate::Type::U32 => {
                        gl::VertexAttribIPointer(
                            i as u32,
                            attr.attr_count() as _,
                            attr.ty().into(),
                            frame_size as _,
                            offset as _,
                        );
                    }
                    crate::Type::F16 | crate::Type::F32 | crate::Type::F64 => {
                        gl::VertexAttribPointer(
                            i as u32,
                            attr.attr_count() as _,
                            attr.ty().into(),
                            gl::FALSE,
                            frame_size as _,
                            offset as _,
                        );
                    }
                }
                panic_if_error();
            }
            // let size = vbuffer.size() / frame_size * attr.size();
            offset += attr.size() as u32;
        }
        Self {
            vao,
            vbo,
            ibo,
            layout: layout.shader_layout,
        }
    }
    pub fn bind<'a, 'b, S>(
        &'a self,
        active_shader: &'b ActiveShader<'a, S>,
    ) -> BoundMeshBuffer<'a, 'b, S> {
        debug_assert!(
            active_shader.layout == self.layout,
            "{:?} != {:?}",
            active_shader.layout,
            self.layout
        );

        unsafe {
            gl::BindVertexArray(self.vao.get());
            panic_if_error();
        }
        BoundMeshBuffer { active_shader }
    }
}

pub struct MeshParams {
    indices_ptr: *const std::os::raw::c_void,
    elements_count: i32,
}

impl MeshParams {
    pub fn new(indices_offset: usize, elements_count: usize) -> Self {
        Self {
            indices_ptr: unsafe {
                std::ptr::null::<std::os::raw::c_void>()
                    .add(indices_offset * std::mem::size_of::<u32>())
            },
            elements_count: elements_count as i32,
        }
    }
}

pub struct MeshSet<T> {
    mesh_buffer: MeshBuffer,
    meshes: Box<[Mesh<T>]>,
}
impl<T> MeshSet<T> {
    pub fn new(
        vertices: &[u8],
        indices: &[u32],
        meshes: Box<[Mesh<T>]>,
        layout: &'static AttribLayout,
    ) -> Self {
        Self {
            mesh_buffer: MeshBuffer::new(vertices, indices, layout),
            meshes,
        }
    }
    pub fn len(&self) -> usize {
        self.meshes.len()
    }
    pub fn bind<'a, 'b, S>(
        &'a self,
        active_shader: &'b ActiveShader<'a, S>,
    ) -> BoundMeshSet<'a, 'b, S, T> {
        BoundMeshSet {
            meshes: &self.meshes,
            bound_mesh_buffer: self.mesh_buffer.bind(active_shader),
        }
    }
}

pub struct MeshIndex(pub usize);

pub struct BoundMeshBuffer<'a, 'b, S> {
    pub active_shader: &'b ActiveShader<'a, S>,
}
impl<'a, 'b, S> BoundMeshBuffer<'a, 'b, S> {
    pub fn set_uniform<U, V>(&self, uniform: U, val: V)
    where
        ActiveShader<'a, S>: SetUniform<U, V>,
    {
        self.active_shader.set(uniform, val)
    }
    pub fn render(&self, mesh: &MeshParams) {
        unsafe { crate::draw(mesh.indices_ptr, mesh.elements_count) }
    }
}

pub struct BoundMeshSet<'a, 'b, S, T> {
    pub meshes: &'a Box<[Mesh<T>]>,
    pub bound_mesh_buffer: BoundMeshBuffer<'a, 'b, S>,
}
impl<'a, 'b, S, T> BoundMeshSet<'a, 'b, S, T> {
    pub fn draw_mesh(&self, idx: usize) {
        let mesh = &self.meshes[idx];
        self.bound_mesh_buffer.render(&mesh.mesh_params)
    }
    pub fn get(&self, mesh: usize) -> &T {
        &self.meshes[mesh].extra
    }
    pub fn len(&self) -> usize {
        self.meshes.len()
    }
}
