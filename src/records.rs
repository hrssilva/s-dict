//! An automaton's record bytes: owned, or a range of an immutable image
//! shared by every block of a dictionary loaded without copying (a file
//! read into one allocation, or memory-mapped). Reads see a plain slice
//! either way; the first write to a shared range copies it out (copy on
//! write), so a loaded dictionary stays mutable, block by block.
//!
//! Reads are the automaton's hot path, so the bytes are kept as a raw view
//! (pointer and length) that is always current, rather than an enum
//! matched on every access: that match measured +16% insertion time and
//! +8% `get_id` with owned records alone.

use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

/// The bytes a zero-copy load borrows from. Only these sources: each
/// returns the same bytes, at the same address, for as long as it lives,
/// which the view in [`Records`] and the validation done at load time
/// rely on (an arbitrary `AsRef<[u8]>` need not).
pub(crate) enum Image {
    Heap(Arc<[u8]>),
    #[cfg(feature = "mmap")]
    Map(memmap2::Mmap),
}

impl Image {
    #[inline]
    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            Image::Heap(bytes) => bytes,
            #[cfg(feature = "mmap")]
            Image::Map(map) => map,
        }
    }
}

pub(crate) struct Records {
    /// The current bytes: `owned`'s contents when `image` is `None`, else
    /// a range of `image.bytes()`. Kept current by every mutation (all go
    /// through [`RecordsMut`], which re-points the view when dropped).
    ptr: NonNull<u8>,
    len: usize,
    owned: Vec<u8>,
    image: Option<Arc<Image>>,
}

// SAFETY: `Records` owns exactly a `Vec<u8>` and an `Arc<Image>` (both
// `Send + Sync`: `Image` holds an `Arc<[u8]>` or an `Mmap`); `ptr` only
// views bytes one of them owns and is never written through.
unsafe impl Send for Records {}
// SAFETY: as above; `&Records` only reads.
unsafe impl Sync for Records {}

impl Records {
    fn owned(owned: Vec<u8>) -> Records {
        let mut records = Records {
            ptr: NonNull::dangling(),
            len: 0,
            owned,
            image: None,
        };
        records.point_at_owned();
        records
    }

    /// `image.bytes()[range]`.
    pub(crate) fn shared(image: &Arc<Image>, range: std::ops::Range<usize>) -> Records {
        let slice = &image.bytes()[range];
        Records {
            ptr: NonNull::from(slice).cast(),
            len: slice.len(),
            owned: Vec::new(),
            image: Some(image.clone()),
        }
    }

    fn point_at_owned(&mut self) {
        self.ptr = NonNull::new(self.owned.as_mut_ptr()).expect("a Vec's pointer is non-null");
        self.len = self.owned.len();
    }

    /// The bytes as an owned, growable buffer (copying a shared range out
    /// first), behind a guard that re-points the view when dropped.
    #[inline(always)]
    pub(crate) fn to_mut(&mut self) -> RecordsMut<'_> {
        if self.image.is_some() {
            self.copy_out();
        }
        RecordsMut(self)
    }

    #[cold]
    #[inline(never)]
    fn copy_out(&mut self) {
        self.owned = self.to_vec();
        self.image = None;
        self.point_at_owned();
    }

    /// Trims an owned buffer's spare capacity (a shared range has none).
    pub(crate) fn shrink_to_fit(&mut self) {
        if self.image.is_none() {
            self.to_mut().shrink_to_fit();
        }
    }
}

/// Mutable access to [`Records`]' owned buffer; re-points the view on drop.
pub(crate) struct RecordsMut<'a>(&'a mut Records);

impl Deref for RecordsMut<'_> {
    type Target = Vec<u8>;

    #[inline(always)]
    fn deref(&self) -> &Vec<u8> {
        &self.0.owned
    }
}

impl DerefMut for RecordsMut<'_> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Vec<u8> {
        &mut self.0.owned
    }
}

impl Drop for RecordsMut<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.0.point_at_owned();
    }
}

impl From<Vec<u8>> for Records {
    fn from(owned: Vec<u8>) -> Self {
        Records::owned(owned)
    }
}

impl Default for Records {
    fn default() -> Self {
        Records::owned(Vec::new())
    }
}

impl Clone for Records {
    fn clone(&self) -> Self {
        match &self.image {
            None => Records::owned(self.owned.clone()),
            // The same range of the same image.
            Some(image) => Records {
                ptr: self.ptr,
                len: self.len,
                owned: Vec::new(),
                image: Some(image.clone()),
            },
        }
    }
}

impl Deref for Records {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr`/`len` always describe live, initialized bytes that
        // `self` keeps alive and nothing mutates while `&self` exists:
        // either `owned`'s contents (re-pointed after every mutation by
        // `RecordsMut::drop`, the only way to mutate `owned`; moving a
        // `Vec` doesn't move its buffer), or a range of `image`'s bytes,
        // which `Image` guarantees stay at the same address, unchanged, for
        // as long as the `Arc` held here lives.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl std::fmt::Debug for Records {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = if self.image.is_none() { "owned" } else { "shared" };
        write!(f, "Records({} bytes, {kind})", self.len)
    }
}
