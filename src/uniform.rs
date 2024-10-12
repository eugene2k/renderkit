use crate::__sealed;
use crate::math::{Mat4, Vec3};
use std::marker::PhantomData;
pub trait UniformStore<T: UniformTy> {
    fn get_store(&self) -> &[UniformLocation];
}
impl<T: UniformStore<U>, U: UniformTy> Uniform<T, U> {
    pub const fn new(idx: usize) -> Self {
        Self {
            idx,
            _phantom_t: std::marker::PhantomData,
            _phantom_u: std::marker::PhantomData,
        }
    }
}
pub trait UniformTy: super::__sealed::Sealed {}
pub trait Set<U> {
    fn set(loc: UniformLocation, val: Self);
}
trait SetArray: Sized {
    unsafe fn set(loc: UniformLocation, val: *const Self, size: usize);
}
pub trait Sampler: UniformTy {
    const GL_TEXTURE_TYPE: u32;
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct UniformLocation(pub(crate) i32);
impl UniformLocation {
    pub const fn new(location: i32) -> Self {
        Self(location)
    }
    pub const fn is_valid(&self) -> bool {
        self.0 != -1
    }
}
impl std::convert::Into<i32> for UniformLocation {
    fn into(self) -> i32 {
        self.0
    }
}

pub struct Uniform<T, U: UniformTy> {
    pub(crate) idx: usize,
    _phantom_t: std::marker::PhantomData<T>,
    _phantom_u: std::marker::PhantomData<U>,
}

pub struct SamplerUniform<S, T: UniformTy> {
    pub(crate) uniform: Uniform<S, T>,
    pub(crate) texture_unit: u32,
    _store: PhantomData<S>,
}
impl<S, T> SamplerUniform<S, T>
where
    S: UniformStore<T>,
    T: UniformTy,
{
    pub const fn new(uniform_idx: usize, texture_unit: u32) -> Self {
        Self {
            uniform: Uniform::new(uniform_idx),
            texture_unit,
            _store: PhantomData,
        }
    }
}

macro_rules! declare_uniform_types {
    // ($(&$ty_name:ident : $ty:ty : $func:ident($($arg:expr),*): |$ident:ident| $stmt:stmt),*$(,)?) => {
    //     declare_uniform_types!{$((&) $ty_name: $ty : $func($($arg),*): |$ident| $stmt),*$(,)?}
    // };
    // ($($ty_name:ident : $ty:ty : $func:ident($($arg:expr),*): |$ident:ident| $stmt:stmt),*$(,)?) => {
    //     declare_uniform_types!{$(() $ty_name: $ty : $func($($arg),*): |$ident| $stmt),*}
    // };
    ($($ty_name:ident : $ty:ty : $func:ident($($arg:expr),*): |$ident:ident| $stmt:stmt),*$(,)?) => {
        $(
            #[repr(transparent)]
            pub struct $ty_name {
                inner: $ty
            }
            impl std::convert::From<$ty> for $ty_name {
                fn from(val: $ty) -> Self {
                    Self {
                        inner: val
                    }
                }
            }
            impl<'a> std::convert::From<&'a $ty> for &'a $ty_name {
                fn from(val: &'a $ty) -> Self {
                    unsafe {std::mem::transmute(val)}
                }
            }
            impl UniformTy for $ty_name {}
            impl Set<$ty_name> for $ty_name {
                fn set(location: UniformLocation, value: Self) {
                    let map = |$ident: $ty| { $stmt };
                    unsafe { gl::$func(location.into(), 1, $($arg,)* map(value.inner)) }
                }
            }
            impl<const N: usize> UniformTy for [$ty_name; N] {}
            impl<'a, const N: usize, const M: usize> Set<[$ty_name; N]> for &'a [$ty_name; M] where [$ty_name; N-M]: {
                fn set(location: UniformLocation, value: Self) {
                    unsafe { gl::$func(location.into(), 1, $($arg,)* value.as_ptr() as _) }
                }
            }
            impl __sealed::Sealed for $ty_name {}
            impl<const N: usize> __sealed::Sealed for [$ty_name; N] {}
        )*
    };
}
// declare_uniform_types! {
//     Vec3: [f32; 3]: Uniform3fv(): |val| val.as_ptr(),
//     Mat4: [f32; 16]: UniformMatrix4fv(false as u8): |val| val.as_ptr(),
// }

impl UniformTy for Vec3 {}
impl Set<Vec3> for Vec3 {
    fn set(location: UniformLocation, value: Self) {
        unsafe { gl::Uniform3fv(location.into(), 1, value.as_ref().as_ptr()) }
    }
}
impl<const N: usize> UniformTy for [Vec3; N] {}
impl<'a, const N: usize, const M: usize> Set<[Vec3; N]> for &'a [Vec3; M]
where
    [Vec3; N - M]:,
{
    fn set(location: UniformLocation, value: Self) {
        unsafe { gl::Uniform3fv(location.into(), 1, value.as_ptr() as _) }
    }
}
impl __sealed::Sealed for Vec3 {}
impl<const N: usize> __sealed::Sealed for [Vec3; N] {}
// #[derive(Debug)]
// #[repr(transparent)]
// pub struct Mat4 {
//     inner: [[f32; 4]; 4],
// }
// impl std::convert::From<[[f32; 4]; 4]> for Mat4 {
//     fn from(val: [[f32; 4]; 4]) -> Self {
//         Self { inner: val }
//     }
// }
// impl<'a> std::convert::From<&'a [f32; 16]> for &'a Mat4 {
//     fn from(val: &'a [f32; 16]) -> Self {
//         unsafe { std::mem::transmute(val) }
//     }
// }
// impl<'a> std::convert::From<&'a [[f32; 4]; 4]> for &'a Mat4 {
//     fn from(val: &'a [[f32; 4]; 4]) -> Self {
//         unsafe { std::mem::transmute(val) }
//     }
// }

impl __sealed::Sealed for Mat4 {}
impl UniformTy for Mat4 {}

impl<const N: usize> UniformTy for [Mat4; N] {}
impl<const N: usize> __sealed::Sealed for [Mat4; N] {}

impl SetArray for Mat4 {
    unsafe fn set(loc: UniformLocation, val: *const Self, size: usize) {
        gl::UniformMatrix4fv(loc.into(), size as i32, 0, val as _)
    }
}

impl<'a, T> Set<T> for &'a T
where
    T: SetArray,
{
    fn set(loc: UniformLocation, val: Self) {
        unsafe { <T as SetArray>::set(loc, val as *const T as _, 1) }
    }
}

impl<'a, T, const N: usize, const M: usize> Set<[T; N]> for &'a [T; M]
where
    [T; N - M]:,
    T: SetArray,
{
    fn set(loc: UniformLocation, val: Self) {
        unsafe { T::set(loc, val.as_ptr(), M) }
    }
}
impl<'a, T, const N: usize> Set<[T; N]> for &'a [T]
where
    T: SetArray,
{
    fn set(loc: UniformLocation, val: Self) {
        debug_assert!(
            val.len() <= N,
            "Max num of elements allowed: {N}, the slice contains {}",
            val.len()
        ); //NOTE: should debug_assert be used or standard assert?
        unsafe { T::set(loc, val.as_ptr(), val.len()) }
    }
}

macro_rules! declare_sampler_types {
    ($($ty_name:ident = $gl_tex_type: expr;)*) => {
        $(
            pub struct $ty_name;
            impl __sealed::Sealed for $ty_name {}
            impl UniformTy for $ty_name {}
            impl Sampler for $ty_name {
                const GL_TEXTURE_TYPE: u32 = $gl_tex_type;
            }
        )*
    };
}
declare_sampler_types! {
    Sampler2D = gl::TEXTURE_2D;
    Sampler2DArray = gl::TEXTURE_2D_ARRAY;
    SamplerCube = gl::TEXTURE_CUBE_MAP;
}
