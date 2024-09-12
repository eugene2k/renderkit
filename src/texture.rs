use std::num::NonZero;

use format::{AsPtr, DataType, InternalFormat, PixelFormat, ValidDataType, ValidPixelFormat};

use crate::{Sampler, Sampler2D, Sampler2DArray, Type};

use super::panic_if_error;
// pub struct TextureRef<'a> {
//     pub(crate) width: usize,
//     pub(crate) height: usize,
//     pub(crate) format: BaseTextureFormat,
//     pub(crate) data: &'a [u8],
// }
// impl<'a> TextureRef<'a> {
//     pub fn new(width: usize, height: usize, format: BaseTextureFormat, data: &'a [u8]) -> Self {
//         let mut buffer = crate::RawBuffer::<crate::ByteAligned>::new(data.len());
//         buffer.extend(data);
//         Self {
//             width,
//             height,
//             format,
//             data,
//         }
//     }
// }
// impl<'a> std::convert::AsRef<TextureRef<'a>> for TextureRef<'a> {
//     fn as_ref(&self) -> &TextureRef<'a> {
//         self
//     }
// }

#[repr(u8)]
pub enum TextureType {
    Albedo,
    Normal,
    Roughness,
}

pub mod format {

    pub trait FormatsTuple {
        const N: usize;
        fn format_values() -> [u32; Self::N];
    }
    impl<T> FormatsTuple for (T,)
    where
        T: InternalFormat,
    {
        const N: usize = 1;

        fn format_values() -> [u32; Self::N] {
            [T::VALUE]
        }
    }

    impl<T, U> FormatsTuple for (T, U)
    where
        T: InternalFormat,
        U: InternalFormat,
    {
        const N: usize = 2;

        fn format_values() -> [u32; Self::N] {
            [T::VALUE, U::VALUE]
        }
    }
    impl<T, U, V> FormatsTuple for (T, U, V)
    where
        T: InternalFormat,
        U: InternalFormat,
        V: InternalFormat,
    {
        const N: usize = 3;

        fn format_values() -> [u32; Self::N] {
            [T::VALUE, U::VALUE, V::VALUE]
        }
    }
    macro_rules! value_impl {
        ($($ident:ident = $expr:expr),*) => {
            pub trait Value {
                const VALUE: u32;
            }
            $(
                pub struct $ident;
                impl Value for $ident {
                    const VALUE: u32 = $expr;
                }
            )*
        };
    }
    macro_rules! decl_marker_traits {
        ($($trait:ident$(<$($t:ident),*>)?),*) => {
            $(
                pub trait $trait$(<$($t),*>)?: Value {}
            )*
        };
    }
    macro_rules! impl_marker_traits {
        ($($trait:path:$($ident:ident),*;)*) => {
            $(
                $(
                    impl $trait for $ident {}
                )*
            )*
        };
    }
    value_impl! {
        Red = gl::RED,
        RedGreen = gl::RG,
        RedGreenBlue = gl::RGB,
        RedGreenBlueAlpha = gl::RGBA,

        DepthComponent = gl::DEPTH_COMPONENT,
        DepthStencil = gl::DEPTH_STENCIL,

        BlueGreenRed = gl::BGR,
        BlueGreenRedAlpha = gl::BGRA,
        RedInt = gl::RED_INTEGER,
        RedGreenInt = gl::RG_INTEGER,
        RedGreenBlueInt = gl::RGB_INTEGER,
        BlueGreenRedInt = gl::BGR_INTEGER,
        RedGreenBlueAlphaInt = gl::RGBA_INTEGER,
        BlueGreenRedAlphaInt = gl::BGRA_INTEGER,
        StencilIndex = gl::STENCIL_INDEX,
        Byte = gl::BYTE,
        UnsignedByte = gl::UNSIGNED_BYTE,
        Short = gl::SHORT,
        UnsignedShort = gl::UNSIGNED_SHORT,
        Int = gl::INT,
        UnsignedInt = gl::UNSIGNED_INT,
        Float = gl::FLOAT
    }
    decl_marker_traits!(InternalFormat, PixelFormat, DataType);
    impl_marker_traits! {
        InternalFormat:         Red, RedGreen, RedGreenBlue, RedGreenBlueAlpha, DepthComponent, DepthStencil;
        PixelFormat:            Red, RedGreen, RedGreenBlue, RedGreenBlueAlpha, RedInt, RedGreenInt, RedGreenBlueInt,
                                RedGreenBlueAlphaInt, BlueGreenRed, BlueGreenRedAlpha, BlueGreenRedInt, BlueGreenRedAlphaInt, DepthComponent, DepthStencil;
        DataType:               Byte, UnsignedByte, Short, UnsignedShort, Int, UnsignedInt, Float;

        ValidPixelFormat<Red>:                  RedGreen, RedGreenBlue, RedInt, RedGreenBlueAlpha, RedGreenInt, RedGreenBlueInt,
                                                RedGreenBlueAlphaInt, BlueGreenRed, BlueGreenRedAlpha, BlueGreenRedInt, BlueGreenRedAlphaInt;
        ValidPixelFormat<RedGreen>:             Red, RedGreenBlue, RedInt, RedGreenBlueAlpha, RedGreenInt, RedGreenBlueInt,
                                                RedGreenBlueAlphaInt, BlueGreenRed, BlueGreenRedAlpha, BlueGreenRedInt, BlueGreenRedAlphaInt;
        ValidPixelFormat<RedGreenBlue>:         Red, RedGreen, RedInt, RedGreenBlueAlpha, RedGreenInt, RedGreenBlueInt,
                                                RedGreenBlueAlphaInt, BlueGreenRed, BlueGreenRedAlpha, BlueGreenRedInt, BlueGreenRedAlphaInt;
        ValidPixelFormat<RedGreenBlueAlpha>:    Red, RedGreen, RedGreenBlue, RedInt, RedGreenInt, RedGreenBlueInt,
                                                RedGreenBlueAlphaInt, BlueGreenRed, BlueGreenRedAlpha, BlueGreenRedInt, BlueGreenRedAlphaInt;
        ValidPixelFormat<BlueGreenRed>: RedGreenBlue;
    }
    pub trait ValidDataType<T: DataType> {}
    impl<T> ValidDataType<Byte> for T where T: PixelFormat {}
    impl<T> ValidDataType<Short> for T where T: PixelFormat {}
    impl<T> ValidDataType<Int> for T where T: PixelFormat {}
    impl<T> ValidDataType<Float> for T where T: PixelFormat {}
    impl<T> ValidDataType<UnsignedByte> for T where T: PixelFormat {}
    impl<T> ValidDataType<UnsignedShort> for T where T: PixelFormat {}
    impl<T> ValidDataType<UnsignedInt> for T where T: PixelFormat {}
    pub trait ValidPixelFormat<T: PixelFormat> {}
    impl<T> ValidPixelFormat<T> for T where T: InternalFormat + PixelFormat {}
    pub trait AsPtr {
        fn as_ptr(&self) -> *const u8;
    }
    impl<T> AsPtr for [T] {
        fn as_ptr(&self) -> *const u8 {
            <[T]>::as_ptr(self).cast()
        }
    }
}

#[derive(Clone, Copy)]
enum BitDepth {
    Eight,
    Sixteen,
    TwentyFour,
    ThirtyTwo,
}

pub struct TextureArray {
    width: u32,
    height: u32,
    layers: u32,
    ty: super::Type,
    buffer: crate::RawBuffer<crate::ByteAligned>,
    uninitialized_layers: u32,
}

impl TextureArray {
    pub fn new(width: u32, height: u32, layers: u32, ty: super::Type) -> Self {
        // TODO: do I need to have the buffer aligned to ty.size()?
        let buffer = crate::RawBuffer::<crate::ByteAligned>::new(
            (width * height * layers) as usize * ty.size() as usize,
        );
        Self {
            width,
            height,
            ty,
            buffer,
            layers,
            uninitialized_layers: layers,
        }
    }
    pub fn add_texture(&mut self, data: &[u8]) {
        if self.uninitialized_layers == 0 {
            panic!("No more space for textures in the array!")
        }
        let expected_size = (self.width * self.height) as usize * self.ty.size() as usize;
        if data.len() != expected_size {
            panic!(
                "Expected texture data to be of size {expected_size}, got {}",
                data.len()
            )
        }
        self.buffer.extend(data)
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn layers(&self) -> u32 {
        self.layers
    }
    pub fn ty(&self) -> Type {
        self.ty
    }
}
// #[derive(Clone, Copy)]

pub struct Texture2DArray {
    texture_handle: TextureHandle,
    backup_texture_handle: TextureHandle,
    width: u32,
    height: u32,
    layers: u32,
    capacity: u32,
    ty: Type,
    mip_level_count: u32,
    tex_fmt: u32,
    pix_fmt: u32,
}
impl Texture2DArray {
    pub fn get_handle(&self) -> TextureHandle {
        self.texture_handle
    }
    pub fn new_with_capacity<T, P>(
        width: u32,
        height: u32,
        capacity: NonZero<u32>,
        ty: Type,
        mip_level_count: u32,
    ) -> Self
    where
        T: InternalFormat,
        P: PixelFormat,
    {
        unsafe {
            let mut texture = std::mem::MaybeUninit::<[NonZero<u32>; 2]>::uninit();
            gl::GenTextures(2, texture.as_mut_ptr() as _);
            panic_if_error();
            let [texture_handle, backup_texture_handle] = texture.assume_init();
            gl::BindTexture(gl::TEXTURE_2D_ARRAY, texture_handle.into());
            panic_if_error();
            gl::TexImage3D(
                gl::TEXTURE_2D_ARRAY,
                mip_level_count as _,
                T::VALUE as _,
                width as _,
                height as _,
                u32::from(capacity) as _,
                0,
                gl::RED,
                ty as _,
                std::ptr::null(),
            );
            panic_if_error();
            Self {
                texture_handle: TextureHandle {
                    handle: texture_handle,
                },
                backup_texture_handle: TextureHandle {
                    handle: backup_texture_handle,
                },
                width,
                height,
                layers: 0,
                capacity: capacity.into(),
                ty,
                mip_level_count,
                pix_fmt: P::VALUE,
                tex_fmt: T::VALUE,
            }
        }

        // Self::new_internal(width, height, capacity, format, ty)
    }

    pub fn resize(&mut self, new_capacity: u32) {
        std::mem::swap(&mut self.backup_texture_handle, &mut self.texture_handle);
        unsafe {
            gl::BindTexture(gl::TEXTURE_2D_ARRAY, self.texture_handle.handle.into());
            panic_if_error();
            gl::TexImage3D(
                gl::TEXTURE_2D_ARRAY,
                self.mip_level_count as _,
                self.tex_fmt as _,
                self.width as _,
                self.height as _,
                new_capacity as _,
                0,
                self.pix_fmt as _,
                self.ty as _,
                std::ptr::null(),
            );
            panic_if_error();
            gl::BindTexture(gl::READ_BUFFER, self.backup_texture_handle.handle.into());
            panic_if_error();
            gl::CopyTexSubImage3D(
                gl::TEXTURE_2D_ARRAY,
                self.mip_level_count as _,
                0,
                0,
                0,
                0,
                0,
                self.width as _,
                self.height as _,
            );
            panic_if_error();
        }
    }
    pub fn try_push(&mut self, texture: &[u8]) -> bool {
        if self.capacity < self.layers {
            unsafe {
                gl::BindTexture(gl::TEXTURE_2D_ARRAY, self.texture_handle.handle.into());
                panic_if_error();
                gl::TexSubImage3D(
                    gl::TEXTURE_2D_ARRAY,
                    0,
                    0,
                    0,
                    self.layers as _,
                    self.width as _,
                    self.height as _,
                    1,
                    self.pix_fmt as _,
                    self.ty as _,
                    texture.as_ptr() as _,
                )
            }
            self.layers += 1;
            true
        } else {
            false
        }
    }

    // fn new_internal(
    //     width: u32,
    //     height: u32,
    //     layers: u32,
    //     format: TextureFormat,
    //     ty: Type,
    //     ptr: *const u8,
    // ) -> Self {
    //     unsafe {
    //         let mut texture = std::mem::MaybeUninit::<u32>::uninit();
    //         gl::GenTextures(1, texture.as_mut_ptr());
    //         panic_if_error();
    //         let texture_handle = texture.assume_init();
    //         gl::BindTexture(gl::TEXTURE_2D_ARRAY, texture_handle);
    //         panic_if_error();
    //         gl::TexImage3D(
    //             gl::TEXTURE_2D_ARRAY,
    //             0,
    //             format as _,
    //             width as _,
    //             height as _,
    //             layers as _,
    //             0,
    //             gl::RGB,
    //             ty as _,
    //             ptr.cast(),
    //         );
    //         panic_if_error();
    //         gl::GenerateMipmap(gl::TEXTURE_2D);
    //         panic_if_error();
    //         Self {
    //             texture_handle: TextureHandle {
    //                 handle: texture_handle,
    //             },
    //         }
    //     }
    // }
}
// impl std::convert::From<TextureArray> for Texture2DArray {
//     fn from(value: TextureArray) -> Self {
//         Self::new_internal(
//             value.width,
//             value.height,
//             value.layers,
//             value.format,
//             value.ty,
//             value.buffer.as_ptr(),
//         )
//     }
// }

// trait TextureDataType {
//     const FORMAT: BaseTextureFormat;
// }

// impl TextureDataType for f32 {
//     const FORMAT: BaseTextureFormat = BaseTextureFormat::;
// }

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Texture2D {
    pub(crate) texture_handle: TextureHandle,
}
impl Texture2D {
    pub fn new<IntFmt, PixFmt, DataFmt, Data>(
        width: u32,
        height: u32,
        pixels: Option<&Data>,
    ) -> Self
    where
        IntFmt: InternalFormat + ValidPixelFormat<PixFmt>,
        PixFmt: PixelFormat + ValidDataType<DataFmt>,
        DataFmt: DataType,
        Data: AsPtr + ?Sized,
    {
        unsafe {
            let mut texture = std::mem::MaybeUninit::<NonZero<u32>>::uninit();
            gl::GenTextures(1, texture.as_mut_ptr() as _);
            panic_if_error();
            let texture_handle = texture.assume_init();
            gl::BindTexture(gl::TEXTURE_2D, texture_handle.into());
            panic_if_error();
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                IntFmt::VALUE as _,
                width as _,
                height as _,
                0,
                PixFmt::VALUE,
                DataFmt::VALUE,
                pixels
                    .map(|r| r.as_ptr())
                    .unwrap_or(std::ptr::null())
                    .cast(),
            );
            panic_if_error();
            gl::GenerateMipmap(gl::TEXTURE_2D);
            panic_if_error();
            Self {
                texture_handle: TextureHandle {
                    handle: texture_handle,
                },
            }
        }
    }
}

impl Texture for Texture2D {
    type SamplerType = Sampler2D;
    fn get_handle(&self) -> TextureHandle {
        self.texture_handle
    }
}
impl Texture for Texture2DArray {
    type SamplerType = Sampler2DArray;
    fn get_handle(&self) -> TextureHandle {
        self.texture_handle
    }
}
pub trait Texture {
    type SamplerType: Sampler;
    fn get_handle(&self) -> TextureHandle;
    // fn bind(&self, texture_unit: u32);
}

#[derive(Debug, Clone, Copy)]
pub struct TextureHandle {
    pub(crate) handle: NonZero<u32>,
}
