use std::{marker::ConstParamTy, num::NonZero};

use super::*;

#[derive(PartialEq, Eq, ConstParamTy)]
#[repr(u32)]
enum BufferKind {
    Array = gl::ARRAY_BUFFER,
    Elements = gl::ELEMENT_ARRAY_BUFFER,
}
struct Buffer<const KIND: BufferKind> {
    buffer_object: NonZero<u32>,
}
impl<const KIND: BufferKind> Buffer<KIND> {
    pub unsafe fn from_buffer_object_unchecked(buffer_object: NonZero<u32>) -> Self {
        Self { buffer_object }
    }
    pub fn new() -> Self {
        let mut buffer = std::mem::MaybeUninit::<NonZero<u32>>::uninit();
        unsafe {
            gl::GenBuffers(1, buffer.as_mut_ptr() as _);
        }
        panic_if_error();
        Self {
            buffer_object: unsafe { buffer.assume_init() },
        }
    }
    pub fn as_buffer_object(&self) -> NonZero<u32> {
        self.buffer_object
    }
    pub fn set_data(&mut self, data: Option<&[u8]>) {
        let (ptr, len) = data
            .map(|data| (data.as_ptr() as _, data.len() as isize))
            .unwrap_or((std::ptr::null() as _, 0isize));
        unsafe {
            gl::BufferData(KIND as _, len, ptr, gl::STATIC_DRAW);
        }
        panic_if_error();
    }
}

#[repr(u32)]
enum BindPoint {
    BufferKind(BufferKind),
    CopyBindPoint(CopyBindPoint),
}
impl Into<u32> for BindPoint {
    fn into(self) -> u32 {
        match self {
            Self::BufferKind(b) => b as u32,
            Self::CopyBindPoint(b) => b as u32,
        }
    }
}

#[repr(u32)]
enum CopyBindPoint {
    CopyReadBuffer = gl::COPY_READ_BUFFER,
    CopyWriteBuffer = gl::COPY_WRITE_BUFFER,
}
struct ArrayObject {
    array_object: NonZero<u32>,
}
impl ArrayObject {
    fn new() -> Self {
        unsafe {
            let mut uninit_vao = std::mem::MaybeUninit::<NonZero<u32>>::uninit();
            gl::GenVertexArrays(1, uninit_vao.as_mut_ptr() as _);
            panic_if_error();
            Self {
                array_object: uninit_vao.assume_init(),
            }
        }
    }
    fn bind<const KIND: BufferKind>(&self, buffer: Buffer<KIND>) {
        unsafe { gl::BindBuffer(KIND as _, buffer.buffer_object.get()) }
    }
    fn bind_copy<const KIND: BufferKind>(&self, bind_point: CopyBindPoint, buffer: Buffer<KIND>) {
        unsafe { gl::BindBuffer(bind_point as _, buffer.buffer_object.get()) }
    }
    fn copy(
        src_target: BindPoint,
        dst_target: BindPoint,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            gl::CopyBufferSubData(
                src_target.into(),
                dst_target.into(),
                src_offset as _,
                dst_offset as _,
                size as _,
            )
        }
        panic_if_error();
    }
}
