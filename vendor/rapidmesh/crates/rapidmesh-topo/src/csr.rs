//! A compact compressed-row adjacency for the variable-degree stars
//! (vertex→tris, vertex→tets, vertex→edges). Offsets + flat data, no per-row
//! allocation, cache-friendly iteration.

/// Compressed adjacency: `row(k)` is the slice of values associated with key `k`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Csr {
    offsets: Vec<u32>,
    data: Vec<u32>,
}

impl Csr {
    /// Build from unordered `(key, value)` pairs over `n_keys` keys. O(n).
    /// Values within a row appear in pair order.
    pub fn from_pairs(n_keys: usize, pairs: &[(u32, u32)]) -> Self {
        let mut offsets = vec![0u32; n_keys + 1];
        for &(k, _) in pairs {
            offsets[k as usize + 1] += 1;
        }
        for i in 0..n_keys {
            offsets[i + 1] += offsets[i];
        }
        let mut data = vec![0u32; pairs.len()];
        let mut cursor = offsets.clone();
        for &(k, v) in pairs {
            let slot = cursor[k as usize];
            data[slot as usize] = v;
            cursor[k as usize] = slot + 1;
        }
        Csr { offsets, data }
    }

    /// The values for key `k`.
    #[inline]
    pub fn row(&self, k: usize) -> &[u32] {
        &self.data[self.offsets[k] as usize..self.offsets[k + 1] as usize]
    }

    /// Number of keys.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_collect_pairs() {
        // keys 0..3; key 0 -> {10,11}, key 1 -> {}, key 2 -> {20}
        let csr = Csr::from_pairs(3, &[(0, 10), (2, 20), (0, 11)]);
        assert_eq!(csr.len(), 3);
        assert_eq!(csr.row(0), &[10, 11]);
        assert_eq!(csr.row(1), &[] as &[u32]);
        assert_eq!(csr.row(2), &[20]);
    }
}
