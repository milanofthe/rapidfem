//! Tets with their face neighbours: the store the Delaunay
//! tetrahedralization and the refinement insert into. A neighbour is kept as
//! `tet << 2 | its face`; a dead tet's slot is reused by the next new one.

use crate::simplex::TET_FACES;
use rustc_hash::FxHashMap;

/// No neighbour (or no tet).
pub(crate) const NONE: u32 = u32::MAX;

#[derive(Default)]
pub(crate) struct Tets {
    pub tets: Vec<[u32; 4]>,
    /// Per tet and face: the neighbour as `tet << 2 | its face`.
    pub nbr: Vec<[u32; 4]>,
    pub alive: Vec<bool>,
    free: Vec<u32>,
    /// Per tet, the last epoch it was marked in.
    pub mark: Vec<u32>,
    pub epoch: u32,
    /// The open edges of a cone, kept between them.
    links: Vec<((u32, u32), u32)>,
}

impl Tets {
    /// The given tets, linked across their shared faces.
    pub fn wired(tets: Vec<[u32; 4]>) -> Tets {
        let n = tets.len();
        let mut t = Tets {
            tets,
            nbr: vec![[NONE; 4]; n],
            alive: vec![true; n],
            mark: vec![0; n],
            ..Tets::default()
        };
        let mut open: FxHashMap<[u32; 3], u32> = FxHashMap::default();
        for i in 0..n as u32 {
            for f in 0..4 {
                let mut k = t.face(i, f);
                k.sort_unstable();
                t.glue(i << 2 | f as u32, open.remove(&k), &mut open, k);
            }
        }
        t
    }

    /// The vertices of face `i` of tet `t` (see [`TET_FACES`](crate::simplex::TET_FACES)).
    pub fn face(&self, t: u32, i: usize) -> [u32; 3] {
        let tv = self.tets[t as usize];
        TET_FACES[i].map(|k| tv[k])
    }

    /// A new epoch to mark tets in.
    pub fn next_epoch(&mut self) -> u32 {
        self.epoch += 1;
        self.epoch
    }

    /// A new tet, without neighbours, in a dead tet's slot where there is one.
    pub fn alloc(&mut self, t: [u32; 4]) -> u32 {
        match self.free.pop() {
            Some(id) => {
                self.tets[id as usize] = t;
                self.nbr[id as usize] = [NONE; 4];
                self.alive[id as usize] = true;
                id
            }
            None => {
                self.tets.push(t);
                self.nbr.push([NONE; 4]);
                self.alive.push(true);
                self.mark.push(0);
                (self.tets.len() - 1) as u32
            }
        }
    }

    /// The tets die; their slots go to the next new ones.
    pub fn kill(&mut self, ts: &[u32]) {
        for &t in ts {
            self.alive[t as usize] = false;
            self.free.push(t);
        }
    }

    /// The point `p` coned to each face `(t, i)` of a cavity's boundary:
    /// each new tet is glued to the tet beyond its face and to its
    /// neighbours in the cone. The new tets are appended to `made`; the
    /// cavity itself stays for the caller to kill.
    pub fn cone(&mut self, boundary: &[(u32, usize)], p: u32, made: &mut Vec<u32>) {
        let mut links = std::mem::take(&mut self.links);
        links.clear();
        for &(t, i) in boundary {
            let outside = self.nbr[t as usize][i];
            let f = self.face(t, i);
            let nt = self.alloc([f[0], f[1], f[2], p]);
            made.push(nt);
            self.nbr[nt as usize][3] = outside;
            if outside != NONE {
                self.nbr[(outside >> 2) as usize][(outside & 3) as usize] = nt << 2 | 3;
            }
            for e in 0..3 {
                let (a, b) = (f[e], f[(e + 1) % 3]);
                // The face of the new tet opposite f's third vertex.
                let here = nt << 2 | ((e + 2) % 3) as u32;
                let key = (a.min(b), a.max(b));
                match links
                    .iter()
                    .position(|l| l.0 == key)
                    .map(|k| links.swap_remove(k).1)
                {
                    Some(there) => {
                        self.nbr[nt as usize][(e + 2) % 3] = there;
                        self.nbr[(there >> 2) as usize][(there & 3) as usize] = here;
                    }
                    None => {
                        links.push((key, here));
                    }
                }
            }
        }
        debug_assert!(links.is_empty(), "the cavity boundary is closed");
        self.links = links;
    }

    /// The tets `fill` put into a hole whose boundary faces (sorted) lead
    /// to the tets in `hole` (`NONE` for none): each face glued to the tet
    /// across the hole's side or to the new one sharing it. Returns the last
    /// new tet.
    pub fn fill(
        &mut self,
        fill: impl IntoIterator<Item = [u32; 4]>,
        hole: &FxHashMap<[u32; 3], u32>,
    ) -> Option<u32> {
        let mut open: FxHashMap<[u32; 3], u32> = FxHashMap::default();
        let mut last = None;
        for t in fill {
            let nt = self.alloc(t);
            last = Some(nt);
            for i in 0..4 {
                let mut f = self.face(nt, i);
                f.sort_unstable();
                let here = nt << 2 | i as u32;
                if let Some(&nb) = hole.get(&f) {
                    self.nbr[nt as usize][i] = nb;
                    if nb != NONE {
                        self.nbr[(nb >> 2) as usize][(nb & 3) as usize] = here;
                    }
                } else {
                    self.glue(here, open.remove(&f), &mut open, f);
                }
            }
        }
        debug_assert!(open.is_empty(), "the filled hole is closed");
        last
    }

    /// Glues the face `here` to `there`, or leaves it open under `key`.
    fn glue(
        &mut self,
        here: u32,
        there: Option<u32>,
        open: &mut FxHashMap<[u32; 3], u32>,
        key: [u32; 3],
    ) {
        match there {
            Some(there) => {
                self.nbr[(here >> 2) as usize][(here & 3) as usize] = there;
                self.nbr[(there >> 2) as usize][(there & 3) as usize] = here;
            }
            None => {
                open.insert(key, here);
            }
        }
    }
}
