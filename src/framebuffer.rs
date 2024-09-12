use std::mem::MaybeUninit;

use crate::{
    panic_if_error, Texture2D,
    __sealed::{Eval, True},
    format::FormatsTuple,
};

pub struct FrameBuffer<const WITH_DEPTH: bool, const N: usize> {
    id: u32,
    targets: [Texture2D; N],
}
impl<const WITH_DEPTH: bool, const N: usize> FrameBuffer<WITH_DEPTH, N> {
    // pub fn bind(&self) -> BoundFrameBuffer<Nil, WITH_DEPTH> {
    //     unsafe {
    //         gl::BindFramebuffer(gl::FRAMEBUFFER, self.id);
    //         panic_if_error();
    //     }
    //     BoundFrameBuffer(Nil)
    // }
    pub fn targets(&self) -> &[Texture2D; N] {
        &self.targets
    }
}
impl<const WITH_DEPTH: bool, const N: usize> super::Bind for FrameBuffer<WITH_DEPTH, N> {
    type BoundObject<'a> = BoundFrameBuffer<Nil, WITH_DEPTH>;

    fn bind<'a>(&'a mut self) -> Self::BoundObject<'a> {
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, self.id);
            panic_if_error();
        }
        BoundFrameBuffer(Nil)
    }
}
impl<const WITH_DEPTH: bool, const N: usize> FrameBuffer<WITH_DEPTH, N>
where
    Eval<{ N > 0 }>: True,
{
    pub fn new<T>(width: u32, height: u32, _format: T) -> Self
    where
        T: crate::format::FormatsTuple,
        [(); T::N]:,
    {
        let id;
        let targets: [Texture2D; N];
        unsafe {
            #[cfg(debug_assertions)]
            {
                let mut max_attachments = MaybeUninit::uninit();
                gl::GetIntegerv(gl::MAX_COLOR_ATTACHMENTS, max_attachments.as_mut_ptr());
                assert!(N <= max_attachments.assume_init() as usize);
            }
            let mut framebuffer = MaybeUninit::uninit();
            gl::GenFramebuffers(1, framebuffer.as_mut_ptr());
            panic_if_error();
            id = framebuffer.assume_init();
            gl::BindFramebuffer(gl::FRAMEBUFFER, id);
            panic_if_error();

            let mut uninit_targets = MaybeUninit::uninit();
            let formats = T::format_values();
            gl::GenTextures(formats.len() as _, uninit_targets.as_mut_ptr() as _);
            panic_if_error();
            targets = uninit_targets.assume_init();
            for i in 0..N {
                gl::BindTexture(gl::TEXTURE_2D, targets[i].texture_handle.handle.into());
                panic_if_error();
                gl::TexImage2D(
                    gl::TEXTURE_2D,
                    0,
                    formats[i] as _,
                    width as _,
                    height as _,
                    0,
                    gl::RGBA,
                    gl::UNSIGNED_BYTE,
                    std::ptr::null(),
                );
                panic_if_error();

                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as _);
                panic_if_error();
                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as _);
                panic_if_error();

                gl::FramebufferTexture2D(
                    gl::FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0 + i as u32,
                    gl::TEXTURE_2D,
                    targets[i].texture_handle.handle.into(),
                    0,
                );
                panic_if_error();
            }
            let mut attachments = [gl::COLOR_ATTACHMENT0; N];
            let mut i = 0;
            for buffer in &mut attachments {
                *buffer += i;
                i += 1
            }
            gl::DrawBuffers(N as i32, attachments.as_ptr());
            if WITH_DEPTH {
                let mut depth_buffer = MaybeUninit::uninit();
                gl::GenRenderbuffers(1, depth_buffer.as_mut_ptr());
                panic_if_error();
                let depth_buffer = depth_buffer.assume_init();
                gl::BindRenderbuffer(gl::RENDERBUFFER, depth_buffer);
                panic_if_error();
                gl::RenderbufferStorage(
                    gl::RENDERBUFFER,
                    gl::DEPTH24_STENCIL8,
                    width as _,
                    height as _,
                );
                panic_if_error();
                gl::FramebufferRenderbuffer(
                    gl::FRAMEBUFFER,
                    gl::DEPTH_STENCIL_ATTACHMENT,
                    gl::RENDERBUFFER,
                    depth_buffer,
                );
                panic_if_error();
            }
            let status = gl::CheckFramebufferStatus(gl::FRAMEBUFFER);
            panic_if_error();
            assert_eq!(status, gl::FRAMEBUFFER_COMPLETE);
        }
        Self { id, targets }
    }
}

pub const DEFAULT_FB: FrameBuffer<true, 0> = FrameBuffer { id: 0, targets: [] };

pub struct BoundFrameBuffer<T, const WITH_DEPTH: bool>(T);
// impl<T: BoundFrameBufferAction> BoundFrameBuffer<T, true> {
//     pub fn enable_depth_test(self) -> BoundFrameBuffer<Enable<T>, true> {
//         BoundFrameBuffer(Enable {
//             var: gl::DEPTH_TEST,
//             preceding_action: self.0,
//         })
//     }
//     pub fn depth_fn(self, func: DepthFunc) -> BoundFrameBuffer<DepthFn<T>, true> {
//         BoundFrameBuffer(DepthFn {
//             func,
//             preceding_action: self.0,
//         })
//     }
// }
impl<T: BoundFrameBufferAction, const WITH_DEPTH: bool> BoundFrameBuffer<T, WITH_DEPTH> {
    // pub fn clear_color(self, color: [f32; 4]) -> BoundFrameBuffer<ClearColor<T>, WITH_DEPTH> {
    //     BoundFrameBuffer(ClearColor {
    //         color,
    //         preceding_action: self.0,
    //     })
    // }
    pub fn clear(self) -> BoundFrameBuffer<Clear<T>, WITH_DEPTH> {
        let mut bits = gl::COLOR_BUFFER_BIT;
        if WITH_DEPTH {
            bits |= gl::DEPTH_BUFFER_BIT;
        }
        BoundFrameBuffer(Clear {
            preceding_action: self.0,
            bits,
        })
    }
    pub fn apply(self) -> PreparedFrameBuffer {
        self.0.action();
        PreparedFrameBuffer(())
    }
}

pub struct PreparedFrameBuffer(());

pub trait BoundFrameBufferAction {
    fn action(self);
}

pub struct Nil;
impl BoundFrameBufferAction for Nil {
    fn action(self) {}
}
pub struct Enable<T> {
    var: u32,
    preceding_action: T,
}
impl<T: BoundFrameBufferAction> BoundFrameBufferAction for Enable<T> {
    fn action(self) {
        self.preceding_action.action();
        unsafe {
            gl::Enable(self.var);
            panic_if_error();
        }
    }
}

#[repr(u32)]
pub enum DepthFn {
    Less = gl::LESS,
    Greater = gl::GREATER,
}
// pub struct DepthFn<T> {
//     func: DepthFunc,
//     preceding_action: T,
// }
// impl<T: BoundFrameBufferAction> BoundFrameBufferAction for DepthFn<T> {
//     fn action(self) {
//         self.preceding_action.action();
//         unsafe {
//             gl::DepthFunc(self.func as _);
//             panic_if_error();
//         }
//     }
// }
// pub struct ClearColor<T> {
//     color: [f32; 4],
//     preceding_action: T,
// }
// impl<T: BoundFrameBufferAction> BoundFrameBufferAction for ClearColor<T> {
//     fn action(self) {
//         self.preceding_action.action();
//         let [r, g, b, a] = self.color;
//         unsafe {
//             gl::ClearColor(r, g, b, a);
//             panic_if_error();
//         }
//     }
// }
pub struct Clear<T> {
    preceding_action: T,
    bits: u32,
}
impl<T: BoundFrameBufferAction> BoundFrameBufferAction for Clear<T> {
    fn action(self) {
        self.preceding_action.action();
        unsafe {
            gl::Clear(self.bits);
            panic_if_error();
        }
    }
}
