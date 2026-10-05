//! A face's bounds in the parameters of its surface: rings of points, each
//! with its parameters (a file's parameter curve, or the point projected),
//! unwrapped round the periods, split at the poles, the rings that wind
//! round a period joined into one that bounds a region, and the others
//! moved into its turn.
//!
//! A point at a pole has no angle of its own: it is split in two, one with
//! the angle of the point before it, one with that of the point after, the
//! pole's line between them. The two rims of a band each run over a whole
//! period and neither encloses anything: they become one ring along the
//! first rim, across a seam to the second, back along it and across the
//! seam again. A rim alone closes through the pole on the side of the face.

use rapidmesh_exact::vector::{dist, segment_dist2, wrap_near, V2, V3};

/// A point of a face's bounds: what it stands for, where it is in space
/// and in the parameters.
#[derive(Clone, Copy, Debug)]
pub struct Mark<T> {
    pub tag: T,
    pub p: V3,
    pub uv: V2,
}

/// What the joining of a face's rings asks of its caller.
pub struct Join<'a, T> {
    /// The points of a seam strictly between two places of the parameters.
    pub seam: &'a dyn Fn(V2, V2) -> Vec<(V2, V3)>,
    /// The tag of a point the bounds add (a seam's, a pole's).
    pub own: &'a mut dyn FnMut(V3) -> T,
    /// The sign of the other parameter on the face's side of a rim running
    /// `w` turns round parameter `k`.
    pub side: &'a dyn Fn(usize, i64, &[Mark<T>]) -> f64,
    /// The longest step along a pole's line (in the periodic parameter).
    pub pole_step: f64,
    /// Edges inside the face (their points' parameters and places), which a
    /// seam must not cross.
    pub avoid: &'a [Vec<(V2, V3)>],
}

/// A face's bounds: the rings, the outer one, and whether they are its
/// loops as given (no pole split, no seam added).
pub struct Bounds<T> {
    pub rings: Vec<Vec<Mark<T>>>,
    pub outer: usize,
    pub plain: bool,
}

/// The bounds of a face on a surface with `periods` and `poles` (see
/// [`crate::Surface::periods`] and [`crate::Surface::poles`]) whose loops
/// are `rings` (with their parameters as given), a point within `fit` of a
/// pole taken as the pole.
pub fn bounds<T: Copy>(
    periods: [Option<f64>; 2],
    poles: &[(usize, f64, V3)],
    mut rings: Vec<Vec<Mark<T>>>,
    fit: f64,
    join: &mut Join<'_, T>,
) -> Result<Bounds<T>, String> {
    // Periodic parameters unwrapped along each ring (projected points come
    // back within one period). A point at a pole is skipped and then split.
    let pole_at = |p: V3| {
        poles
            .iter()
            .find(|pole| dist(pole.2, p) <= fit)
            .map(|pole| (pole.0, pole.1))
    };
    let mut plain = true;
    for ring in rings.iter_mut() {
        let at: Vec<Option<(usize, f64)>> = ring.iter().map(|m| pole_at(m.p)).collect();
        let mut prev: Option<V2> = None;
        for i in 0..ring.len() {
            if at[i].is_some() {
                continue;
            }
            if let Some(q) = prev {
                for k in 0..2 {
                    if let Some(period) = periods[k] {
                        ring[i].uv[k] = wrap_near(ring[i].uv[k], q[k], period);
                    }
                }
            }
            prev = Some(ring[i].uv);
        }
        if at.iter().all(Option::is_none) {
            continue;
        }
        plain = false;
        // A ring through a pole that misses its start by a turn (a seam run
        // down to the pole and up again, both times at one angle) takes the
        // turn at the pole, where the angle is free: the points past it
        // move by the turn.
        if let (Some(p0), Some(last)) = (
            at.iter().position(Option::is_some),
            (0..ring.len()).rev().find(|&i| at[i].is_none()),
        ) {
            let first = (0..ring.len()).find(|&i| at[i].is_none()).unwrap_or(0);
            for (k, period) in periods.iter().enumerate() {
                let Some(period) = *period else { continue };
                let turns = ((ring[first].uv[k] - ring[last].uv[k]) / period).round();
                if turns != 0.0 && first < p0 {
                    for i in p0 + 1..ring.len() {
                        if at[i].is_none() {
                            ring[i].uv[k] += turns * period;
                        }
                    }
                }
            }
        }
        if at.iter().all(Option::is_some) {
            return Err("a bound lies in a pole".into());
        }
        let n = ring.len();
        let mut split = Vec::with_capacity(n + 2);
        for i in 0..n {
            let Some((fixed, value)) = at[i] else {
                split.push(ring[i]);
                continue;
            };
            let free = |step: usize| {
                let j = (1..n)
                    .map(|d| (i + step * d) % n)
                    .find(|&j| at[j].is_none())
                    .unwrap_or(i);
                let mut q = ring[j].uv;
                q[fixed] = value;
                q
            };
            let (from, to) = (free(n - 1), free(1));
            split.push(Mark {
                uv: from,
                ..ring[i]
            });
            if from != to {
                split.push(Mark { uv: to, ..ring[i] });
            }
        }
        *ring = split;
    }
    let sizes: Vec<usize> = rings.iter().map(Vec::len).collect();
    join_windings(&mut rings, periods, poles, join)?;
    plain &= rings.iter().map(Vec::len).eq(sizes.iter().copied());
    // A side of a ring along a pole (one point in space) takes points as
    // closely as `pole_step`: facets fan into the pole from next to each
    // other, not from across the face.
    for ring in rings.iter_mut() {
        let n = ring.len();
        let mut dense = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            dense.push(a);
            if a.p == b.p && a.uv != b.uv {
                let span = (b.uv[0] - a.uv[0]).abs().max((b.uv[1] - a.uv[1]).abs());
                let parts = (span / join.pole_step).ceil() as usize;
                for m in 1..parts {
                    let w = m as f64 / parts as f64;
                    dense.push(Mark {
                        uv: [
                            a.uv[0] + w * (b.uv[0] - a.uv[0]),
                            a.uv[1] + w * (b.uv[1] - a.uv[1]),
                        ],
                        ..a
                    });
                }
            }
        }
        *ring = dense;
    }
    let outer = rings
        .iter()
        .enumerate()
        .map(|(i, r)| (i, area(&r.iter().map(|m| m.uv).collect::<Vec<_>>()).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    for k in 0..2 {
        let Some(period) = periods[k] else {
            continue;
        };
        let m0 = mean(&rings[outer], k);
        for (i, r) in rings.iter_mut().enumerate() {
            if i != outer {
                let m = mean(r, k);
                let shift = wrap_near(m, m0, period) - m;
                r.iter_mut().for_each(|q| q.uv[k] += shift);
            }
        }
    }
    Ok(Bounds {
        rings,
        outer,
        plain,
    })
}

/// The mean of parameter `k` over a ring.
pub fn mean<T>(r: &[Mark<T>], k: usize) -> f64 {
    r.iter().map(|q| q.uv[k]).sum::<f64>() / r.len().max(1) as f64
}

/// Twice the signed area of a ring of the parameters.
pub fn area(r: &[V2]) -> f64 {
    (0..r.len())
        .map(|i| {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum()
}

/// Joins the rings that wind once round a period of the surface into one
/// that bounds a region of the parameters (see the module).
fn join_windings<T: Copy>(
    rings: &mut Vec<Vec<Mark<T>>>,
    periods: [Option<f64>; 2],
    poles: &[(usize, f64, V3)],
    join: &mut Join<'_, T>,
) -> Result<(), String> {
    // The turns of each ring round each period, its closing step included.
    let turns = |r: &[Mark<T>], k: usize| -> i64 {
        let Some(period) = periods[k] else {
            return 0;
        };
        // The steps along the ring add up to last - first; the closing one
        // takes the first point next to the last.
        let (a, b) = (r[0].uv[k], r[r.len() - 1].uv[k]);
        ((wrap_near(a, b, period) - a) / period).round() as i64
    };
    let winding: Vec<(usize, usize, i64)> = rings
        .iter()
        .enumerate()
        .flat_map(|(i, r)| (0..2).map(move |k| (i, k, turns(r, k))))
        .filter(|&(_, _, w)| w != 0)
        .collect();
    let Some(&(_, k, _)) = winding.first() else {
        return Ok(());
    };
    if winding.iter().any(|w| w.1 != k || w.2.abs() != 1) {
        return Err("a bound winds round the surface more than once".into());
    }
    let period = periods[k].unwrap_or(0.0);
    let shift = |q: V2, by: f64| -> V2 {
        let mut q = q;
        q[k] += by;
        q
    };
    // The ring from point `from` on round, ending on the copy of that point
    // a turn on.
    let open = |r: &[Mark<T>], from: usize, w: i64| -> Vec<Mark<T>> {
        let n = r.len();
        let turn = w as f64 * period;
        (0..=n)
            .map(|t| {
                let i = from + t;
                if i < n {
                    r[i]
                } else {
                    Mark {
                        uv: shift(r[i - n].uv, turn),
                        ..r[i - n]
                    }
                }
            })
            .collect()
    };
    let j = 1 - k;
    let seam_marks = |a: V2, b: V2, own: &mut dyn FnMut(V3) -> T| -> Vec<Mark<T>> {
        (join.seam)(a, b)
            .into_iter()
            .map(|(uv, p)| Mark { tag: own(p), p, uv })
            .collect()
    };
    let back = |across: &[Mark<T>], w: i64| -> Vec<Mark<T>> {
        across
            .iter()
            .rev()
            .map(|m| Mark {
                uv: shift(m.uv, -(w as f64) * period),
                ..*m
            })
            .collect()
    };
    let joined = match *winding.as_slice() {
        [(a, _, wa), (b, _, wb)] => {
            // The outline runs along one rim and back along the other: a
            // second rim given the same way round is taken the other way.
            let wb = if wa == wb {
                rings[b].reverse();
                -wb
            } else {
                wb
            };
            // The seam runs from a point of the first rim to the point of
            // the second nearest it in space (a rim with a step up in it
            // has points of one angle at two heights): of the starts whose
            // seam crosses no edge of the face, the one farthest from the
            // other points of its edges; failing that, the pair nearest
            // each other. The first rim opens there, the second is moved
            // into its turn.
            let near = |i: usize| {
                (0..rings[b].len())
                    .min_by(|&x, &y| {
                        dist(rings[a][i].p, rings[b][x].p)
                            .total_cmp(&dist(rings[a][i].p, rings[b][y].p))
                    })
                    .unwrap_or(0)
            };
            let far = |i: usize| {
                let m = rings[b][near(i)];
                let mut uv = m.uv;
                uv[k] = wrap_near(uv[k], rings[a][i].uv[k], period);
                (uv, m.p)
            };
            let (start, from) = match seam_start(rings, a, &far, join.avoid, periods) {
                Some(i) => (i, near(i)),
                None => (0..rings[a].len())
                    .flat_map(|i| (0..rings[b].len()).map(move |j| (i, j)))
                    .min_by(|&(i, j), &(x, y)| {
                        dist(rings[a][i].p, rings[b][j].p)
                            .total_cmp(&dist(rings[a][x].p, rings[b][y].p))
                    })
                    .unwrap_or((0, 0)),
            };
            let mut out = open(&rings[a], start, wa);
            let end = out[out.len() - 1].uv[k];
            let by = wrap_near(rings[b][from].uv[k], end, period) - rings[b][from].uv[k];
            let mut moved: Vec<Mark<T>> = rings[b]
                .iter()
                .map(|m| Mark {
                    uv: shift(m.uv, by),
                    ..*m
                })
                .collect();
            // Where the other parameter wraps too (a torus), the second rim
            // goes to the face's side of the first, within a turn: a fillet
            // round a hole is the quarter between its rims, not the rest.
            if let Some(turn) = periods[j] {
                let side = (join.side)(k, wa, &rings[a]);
                let (ma, mb) = (mean(&rings[a], j), mean(&moved, j));
                let ahead = (side * (mb - ma)).rem_euclid(turn);
                let to = ma + side * ahead;
                moved.iter_mut().for_each(|q| q.uv[j] += to - mb);
            }
            // Across the seam to the second rim, round it, and back across
            // the seam a turn on: the same points of the surface.
            let across = seam_marks(out[out.len() - 1].uv, moved[from].uv, join.own);
            let back = back(&across, wa);
            out.extend(across);
            out.extend(open(&moved, from, wb));
            out.extend(back);
            out
        }
        [(a, _, wa)] => {
            let side = (join.side)(k, wa, &rings[a]);
            let m = mean(&rings[a], j);
            let Some(&(_, value, pole)) = poles
                .iter()
                .filter(|p| p.0 == j && (p.1 - m) * side > 0.0)
                .min_by(|x, y| (x.1 - m).abs().total_cmp(&(y.1 - m).abs()))
            else {
                return Err("a bound winds round the surface alone".into());
            };
            // The seam runs from the rim to the pole along the period's
            // other parameter, from the start clear of the face's edges.
            let far = |i: usize| {
                let mut uv = rings[a][i].uv;
                uv[j] = value;
                (uv, pole)
            };
            let start = seam_start(rings, a, &far, join.avoid, periods).unwrap_or(0);
            let mut out = open(&rings[a], start, wa);
            let (first, last) = (out[0].uv, out[out.len() - 1].uv);
            let mut p0 = last;
            p0[j] = value;
            let mut p1 = first;
            p1[j] = value;
            // Along the seam to the pole and back from it a turn on.
            let across = seam_marks(last, p0, join.own);
            let back = back(&across, wa);
            out.extend(across);
            let tag = (join.own)(pole);
            out.extend([
                Mark {
                    tag,
                    p: pole,
                    uv: p0,
                },
                Mark {
                    tag,
                    p: pole,
                    uv: p1,
                },
            ]);
            out.extend(back);
            out
        }
        _ => return Err("more than two bounds wind round the surface".into()),
    };
    // Rims that touch (a bore cut by another) meet where the seam starts:
    // a seam of no length leaves the point there twice, once is enough.
    let mut joined = joined;
    let mut i = 0;
    while joined.len() > 3 && i < joined.len() {
        let n = (i + 1) % joined.len();
        if joined[i].p == joined[n].p && joined[i].uv == joined[n].uv {
            joined.remove(n);
        } else {
            i += 1;
        }
    }
    let mut gone: Vec<usize> = winding.iter().map(|w| w.0).collect();
    gone.sort_unstable();
    for i in gone.into_iter().rev() {
        rings.remove(i);
    }
    rings.push(joined);
    Ok(())
}

/// Where a seam starts on rim `a` of `rings`, running straight in the
/// parameters to `far(i)` from point `i`: of the starts whose seam crosses
/// no side of the rings and no edge `avoid` inside the face, the one
/// farthest in space from every other point of them (a notch in the rim, a
/// hole, an edge inside). None where every seam crosses one.
fn seam_start<T: Copy>(
    rings: &[Vec<Mark<T>>],
    a: usize,
    far: &dyn Fn(usize) -> (V2, V3),
    avoid: &[Vec<(V2, V3)>],
    periods: [Option<f64>; 2],
) -> Option<usize> {
    // Every side, its end next to its start (the closing side of a rim
    // that winds round runs back over the period it unwrapped).
    let mut sides: Vec<[(V2, V3); 2]> = Vec::new();
    for r in rings {
        for i in 0..r.len() {
            let (x, y) = (r[i], r[(i + 1) % r.len()]);
            let mut q = y.uv;
            for k in 0..2 {
                if let Some(per) = periods[k] {
                    q[k] = wrap_near(q[k], x.uv[k], per);
                }
            }
            sides.push([(x.uv, x.p), (q, y.p)]);
        }
    }
    for c in avoid {
        sides.extend(c.windows(2).map(|w| [w[0], w[1]]));
    }
    let n = rings[a].len();
    let stride = n.div_ceil(128).max(1);
    let clear = |i: usize| -> Option<f64> {
        let (s0, p0) = (rings[a][i].uv, rings[a][i].p);
        let (s1, p1) = far(i);
        let mut clearance = f64::INFINITY;
        for &[(x, px), (y, py)] in &sides {
            // The side moved into the seam's turn, and a turn either way.
            let mut base = x;
            for k in 0..2 {
                if let Some(per) = periods[k] {
                    base[k] = wrap_near(x[k], s0[k], per);
                }
            }
            let d = [base[0] - x[0], base[1] - x[1]];
            let touches = |q: V3| q == p0 || q == p1;
            if !touches(px) && !touches(py) {
                for turn in [-1.0, 0.0, 1.0] {
                    let mut sh = d;
                    if let Some(per) = periods[0] {
                        sh[0] += turn * per;
                    }
                    let (xs, ys) = ([x[0] + sh[0], x[1] + sh[1]], [y[0] + sh[0], y[1] + sh[1]]);
                    if crosses(s0, s1, xs, ys) {
                        return None;
                    }
                }
            }
            for q in [px, py] {
                if !touches(q) {
                    clearance = clearance.min(segment_dist2(q, p0, p1).sqrt());
                }
            }
        }
        Some(clearance)
    };
    (0..n)
        .step_by(stride)
        .filter_map(|i| clear(i).map(|c| (i, c)))
        .max_by(|x, y| x.1.total_cmp(&y.1))
        .map(|x| x.0)
}

/// Whether the segments `a b` and `x y` cross (at a point inside both).
fn crosses(a: V2, b: V2, x: V2, y: V2) -> bool {
    let side = |p: V2, q: V2, r: V2| (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]);
    let (s1, s2) = (side(a, b, x), side(a, b, y));
    let (s3, s4) = (side(x, y, a), side(x, y, b));
    s1 * s2 < 0.0 && s3 * s4 < 0.0
}
