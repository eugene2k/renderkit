use std;

use std::marker::PhantomData;
use std::mem::size_of;

pub enum ByteAligned {}

pub enum WordAligned {}

pub enum DWordAligned {}

pub enum QWordAligned {}

pub trait Alignment {
    const ALIGN: usize;
}

impl Alignment for ByteAligned {
    const ALIGN: usize = 1;
}

impl Alignment for WordAligned {
    const ALIGN: usize = 2;
}

impl Alignment for DWordAligned {
    const ALIGN: usize = 4;
}

impl Alignment for QWordAligned {
    const ALIGN: usize = 8;
}

pub struct RawBuffer<T: Alignment> {
    ptr: *mut u8,
    size: usize,
    uninit_offset: usize,
    _phantom: PhantomData<T>,
}

impl<T: Alignment> RawBuffer<T> {
    pub fn new(size: usize) -> Self {
        let ptr = unsafe {
            std::alloc::alloc(std::alloc::Layout::from_size_align(size, T::ALIGN).unwrap())
        };
        Self {
            ptr,
            size,
            uninit_offset: 0,
            _phantom: PhantomData,
        }
    }
    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    pub fn size(&self) -> usize {
        self.uninit_offset
    }
    unsafe fn push_internal<U>(&mut self, val: U) {
        if self.uninit_offset == self.size {
            panic!("Buffer is full!")
        }
        let add_to_offset = std::mem::size_of_val(&val);
        std::ptr::write(self.ptr.add(self.uninit_offset) as _, val);
        self.uninit_offset += add_to_offset;
    }
    pub fn push<U>(&mut self, val: U) {
        if std::mem::align_of_val(&val) > T::ALIGN {
            panic!("Value isn't properly aligned!")
        } else if std::mem::align_of_val(&val) < T::ALIGN
            && T::ALIGN % std::mem::size_of_val(&val) != 0
        {
            panic!("Value should be padded before pushing into the buffer!")
        }
        unsafe { self.push_internal(val) }
    }
    // TODO: make it so that extend packs the values that have a smaller alignment into arrays of size_in_bytes % alignment = 0
    pub fn extend<U: IntoIterator>(&mut self, val: U) {
        val.into_iter().for_each(|item| self.push(item));
    }
}

impl<T: Alignment> Drop for RawBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(
                self.ptr,
                std::alloc::Layout::from_size_align_unchecked(self.size, T::ALIGN),
            )
        };
    }
}
