//! Globally deduplicated exact vertex pool.

use rapidmesh_exact::Point3;
use rustc_hash::FxHashMap as HashMap;

/// Vertex pool with exact deduplication, hashed on the correctly rounded
/// coordinates: coincident points round to the same f64 triple whatever
/// their construction, so only points of one triple need the exact test.
#[derive(Default)]
pub struct VertexPool {
    /// The deduplicated vertices.
    pub verts: Vec<Point3>,
    by_approx: HashMap<[u64; 3], Vec<usize>>,
}

impl VertexPool {
    /// Returns the index of `p`, inserting it if no exactly coincident
    /// vertex exists yet.
    pub fn insert(&mut self, p: Point3) -> usize {
        let a = p.approx().expect("valid point");
        // `+ 0.0` folds -0.0 into 0.0, which it equals.
        let key = a.map(|x| (x + 0.0).to_bits());
        let ids = self.by_approx.entry(key).or_default();
        // The same construction again is the common case, and exact
        // coincidence of implicit points needs expansions.
        let same = |i: usize| self.verts[i] == p || self.verts[i].coincides(&p);
        if let Some(&i) = ids.iter().find(|&&i| same(i)) {
            return i;
        }
        let id = self.verts.len();
        ids.push(id);
        self.verts.push(p);
        id
    }
}
