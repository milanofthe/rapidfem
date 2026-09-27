//! The "build from anything" boundary. The complex builders take these tiny
//! traits, so a solver's own planar mesh, our `SurfaceMesh`/`TetMesh` (via the
//! `mesher` feature), or raw external arrays all feed the same code. Indexed
//! accessors return `[u32; N]`, so a `[usize; N]`-indexed mesh adapts without a
//! borrow that would force the source to store `u32`.

/// A triangle mesh (planar or surface) as input to [`crate::TriTopology`].
pub trait TriSource {
    fn n_verts(&self) -> usize;
    fn n_tris(&self) -> usize;
    fn tri(&self, i: usize) -> [u32; 3];
    /// Per-triangle tag (conductor / region). Default: untagged.
    fn tri_tag(&self, _i: usize) -> i64 {
        0
    }
}

/// A tetrahedral mesh as input to [`crate::TetTopology`].
pub trait TetSource {
    fn n_verts(&self) -> usize;
    fn n_tets(&self) -> usize;
    fn tet(&self, i: usize) -> [u32; 4];
}

/// Borrowed-slice triangle source (external arrays, tests). `tags` may be empty.
pub struct Tris<'a> {
    pub tris: &'a [[u32; 3]],
    pub tags: &'a [i64],
    pub n_verts: usize,
}

impl TriSource for Tris<'_> {
    fn n_verts(&self) -> usize {
        self.n_verts
    }
    fn n_tris(&self) -> usize {
        self.tris.len()
    }
    fn tri(&self, i: usize) -> [u32; 3] {
        self.tris[i]
    }
    fn tri_tag(&self, i: usize) -> i64 {
        self.tags.get(i).copied().unwrap_or(0)
    }
}

/// Borrowed-slice tet source (external arrays, tests).
pub struct Tets<'a> {
    pub tets: &'a [[u32; 4]],
    pub n_verts: usize,
}

impl TetSource for Tets<'_> {
    fn n_verts(&self) -> usize {
        self.n_verts
    }
    fn n_tets(&self) -> usize {
        self.tets.len()
    }
    fn tet(&self, i: usize) -> [u32; 4] {
        self.tets[i]
    }
}

impl<'a> Tris<'a> {
    /// Convenience constructor when there are no tags.
    pub fn untagged(tris: &'a [[u32; 3]], n_verts: usize) -> Self {
        Tris {
            tris,
            tags: &[],
            n_verts,
        }
    }
}
