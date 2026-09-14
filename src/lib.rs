use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use csgrs::mesh::polygon::Polygon as CsgPolygon;
use csgrs::mesh::vertex::Vertex as CsgVertex;
use csgrs::mesh::Mesh;
use csgrs::sketch::Sketch as CsgSketch;
use csgrs::traits::CSG;
use manifold_rust::linalg::Vec3;
use manifold_rust::manifold::Manifold;
use manifold_rust::types::{BooleanEngine, Error as ManifoldError, MeshGL64};
use nalgebra::{Point3, Vector3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

/// `csgrs` meshes here carry no per-polygon metadata, hence `()`.
type Solid3 = Mesh<()>;
/// Same, for the 2D sketches that get extruded/revolved/swept into solids.
type Sketch2 = CsgSketch<()>;

/// A lazily-evaluated node in a CSG expression tree.
///
/// Building a [`Solid`] (via `box_(..)`, `.translate(..)`, `base - hole`, ...)
/// only ever grows this tree - no mesh is actually built until
/// [`Solid::to_mesh`] (or something that calls it, like the Python-side
/// `.show()`) forces evaluation. That keeps chains of boolean/transform
/// operations cheap to construct and lets the real work happen once, on
/// demand, over the *whole* tree.
struct Node {
    op: Op,
    /// This node's evaluated shape, if it's worth keeping - see
    /// [`Node::eval`] and [`Solid::mesh`].
    shape: OnceLock<Shape>,
}

enum Op {
    Cuboid { width: f64, length: f64, height: f64 },
    Cylinder { radius: f64, height: f64, segments: usize },
    Sphere { radius: f64, segments: usize, stacks: usize },
    Cone { radius: f64, height: f64, segments: usize },
    Translate { child: Arc<Node>, dx: f64, dy: f64, dz: f64 },
    Rotate { child: Arc<Node>, rx_deg: f64, ry_deg: f64, rz_deg: f64 },
    Scale { child: Arc<Node>, sx: f64, sy: f64, sz: f64 },
    Union { a: Arc<Node>, b: Arc<Node> },
    Difference { a: Arc<Node>, b: Arc<Node> },
    Intersection { a: Arc<Node>, b: Arc<Node> },
    /// A [`Sketch`] linearly extruded along an arbitrary vector.
    ExtrudeVector { sketch: Arc<SketchNode>, dx: f64, dy: f64, dz: f64 },
    /// A [`Sketch`] swept around the Y axis into a surface of revolution.
    Revolve { sketch: Arc<SketchNode>, angle_deg: f64, segments: usize },
    /// A [`Sketch`] duplicated along a 3D path, oriented to its tangent at
    /// each point, with the side walls stitched between consecutive copies.
    Sweep { sketch: Arc<SketchNode>, path: Vec<(f64, f64, f64)> },
}

impl Node {
    fn new(op: Op) -> Arc<Node> {
        Arc::new(Node { op, shape: OnceLock::new() })
    }

    /// Recursively realize this node into a [`Shape`]. The only thing that
    /// can fail is a boolean (see [`boolean`]).
    ///
    /// A node referenced from more than one place - `die` in
    /// `[die.translate(i * 30, 0, 0) for i in range(5)]` - remembers its
    /// result, so a shared subtree is computed once rather than once per
    /// use. A node with a single owner (every intermediate step of a
    /// `body = body - hole` loop) doesn't, so long chains don't hold on to
    /// a copy of every step.
    fn eval(self: &Arc<Node>) -> Result<Shape, String> {
        if let Some(shape) = self.shape.get() {
            return Ok(shape.clone());
        }
        let shape = self.op.eval()?;
        if Arc::strong_count(self) > 1 {
            let _ = self.shape.set(shape.clone());
        }
        Ok(shape)
    }
}

impl Op {
    fn eval(&self) -> Result<Shape, String> {
        Ok(match self {
            Op::Cuboid { width, length, height } => {
                Shape::Mesh(Mesh::cuboid(*width, *length, *height, None))
            }
            Op::Cylinder { radius, height, segments } => {
                Shape::Mesh(Mesh::cylinder(*radius, *height, *segments, None))
            }
            Op::Sphere { radius, segments, stacks } => {
                Shape::Mesh(Mesh::sphere(*radius, *segments, *stacks, None))
            }
            Op::Cone { radius, height, segments } => {
                Shape::Mesh(Mesh::frustum(*radius, 0.0, *height, *segments, None))
            }
            Op::Translate { child, dx, dy, dz } => child.eval()?.translate(*dx, *dy, *dz),
            Op::Rotate { child, rx_deg, ry_deg, rz_deg } => {
                child.eval()?.rotate(*rx_deg, *ry_deg, *rz_deg)
            }
            Op::Scale { child, sx, sy, sz } => child.eval()?.scale(*sx, *sy, *sz),
            Op::Union { a, b } => boolean(a.eval()?, b.eval()?, BoolOp::Union)?,
            Op::Difference { a, b } => boolean(a.eval()?, b.eval()?, BoolOp::Difference)?,
            Op::Intersection { a, b } => boolean(a.eval()?, b.eval()?, BoolOp::Intersection)?,
            Op::ExtrudeVector { sketch, dx, dy, dz } => {
                Shape::Mesh(sketch.eval().extrude_vector(Vector3::new(*dx, *dy, *dz)))
            }
            Op::Revolve { sketch, angle_deg, segments } => Shape::Mesh(
                sketch
                    .eval()
                    .revolve(*angle_deg, *segments)
                    // Bad `segments`/`angle_deg` is rejected up front in the
                    // `revolve()` pymethod below, so this only ever hits on the
                    // lazy path (e.g. after a repr/debug print); fall back to an
                    // empty mesh rather than panicking deep inside `.eval()`.
                    .unwrap_or_else(|_| Solid3::new()),
            ),
            Op::Sweep { sketch, path } => {
                let points: Vec<Point3<f64>> =
                    path.iter().map(|&(x, y, z)| Point3::new(x, y, z)).collect();
                Shape::Mesh(sketch.eval().sweep(&points))
            }
        })
    }
}

/// An evaluated [`Node`]. Primitives and sketch operations come out of
/// csgrs as a `Mesh`; booleans come out of Manifold, and stay a `Manifold`
/// through any further transforms and booleans, so a chain like
/// `body - hole1 - hole2 - ...` never converts back and forth between the
/// two in the middle. [`Shape::into_mesh`] converts once, at the end.
#[derive(Clone)]
enum Shape {
    Mesh(Solid3),
    /// Plus which of Manifold's boolean engines this shape needs.
    Manifold(Manifold, Engine),
}

/// Manifold's `Exact` engine is fast but needs clean input: properly
/// connected, not intersecting itself. Its `Robust` engine takes anything
/// closed but is several times slower. (Its `Auto` mode picks per boolean,
/// but by re-scanning both operands for self-intersections every time -
/// in a `body - hole - hole - ...` chain that's the whole growing body at
/// every step, 15x the cost of the boolean itself.) So the check is made
/// once, when a csgrs mesh is imported, and carried along: an `Exact`
/// boolean of clean shapes gives a clean shape.
#[derive(Clone, Copy, PartialEq)]
enum Engine {
    Exact,
    Robust,
}

impl Shape {
    fn translate(self, dx: f64, dy: f64, dz: f64) -> Shape {
        match self {
            Shape::Mesh(mesh) => Shape::Mesh(mesh.translate(dx, dy, dz)),
            Shape::Manifold(m, e) => Shape::Manifold(m.translate(Vec3::new(dx, dy, dz)), e),
        }
    }

    /// Both libraries rotate about X, then Y, then Z (`Rz * Ry * Rx`).
    fn rotate(self, rx_deg: f64, ry_deg: f64, rz_deg: f64) -> Shape {
        match self {
            Shape::Mesh(mesh) => Shape::Mesh(mesh.rotate(rx_deg, ry_deg, rz_deg)),
            Shape::Manifold(m, e) => Shape::Manifold(m.rotate(rx_deg, ry_deg, rz_deg), e),
        }
    }

    fn scale(self, sx: f64, sy: f64, sz: f64) -> Shape {
        match self {
            Shape::Mesh(mesh) => Shape::Mesh(mesh.scale(sx, sy, sz)),
            Shape::Manifold(m, e) => Shape::Manifold(m.scale(Vec3::new(sx, sy, sz)), e),
        }
    }

    fn polygon_count(&self) -> usize {
        match self {
            Shape::Mesh(mesh) => mesh.polygons.len(),
            Shape::Manifold(m, _) => m.num_tri(),
        }
    }

    fn to_manifold(&self) -> Result<(Cow<'_, Manifold>, Engine), ManifoldError> {
        match self {
            Shape::Mesh(mesh) => to_manifold(mesh).map(|(m, e)| (Cow::Owned(m), e)),
            Shape::Manifold(m, e) => Ok((Cow::Borrowed(m), *e)),
        }
    }

    fn into_mesh(self) -> Solid3 {
        match self {
            Shape::Mesh(mesh) => mesh,
            Shape::Manifold(m, _) => from_manifold(&m),
        }
    }
}

#[derive(Clone, Copy)]
enum BoolOp {
    Union,
    Difference,
    Intersection,
}

/// Largest combined polygon count [`boolean`] will hand to csgrs's own BSP
/// booleans when Manifold can't take the operands. Their BSP build copies
/// the remaining polygons at every level, and convex input (a sphere) makes
/// the tree a chain as deep as the polygon count, so worst-case memory is
/// about n^2/2 polygons at a few hundred bytes each: ~700 MB at this limit,
/// tens of GB for a detailed sphere.
const BSP_FALLBACK_MAX_POLYGONS: usize = 2_000;

/// A 3D boolean. Runs on Manifold (fast, memory linear in the mesh size);
/// only if Manifold rejects an operand (e.g. it isn't a closed surface) does
/// it fall back to csgrs's BSP booleans, and only for small inputs - bigger
/// ones fail with an error instead of eating all the RAM.
fn boolean(a: Shape, b: Shape, op: BoolOp) -> Result<Shape, String> {
    let rejected = match (a.to_manifold(), b.to_manifold()) {
        (Ok((ma, ea)), Ok((mb, eb))) => {
            let run = |engine: Engine| {
                let engine = match engine {
                    Engine::Exact => BooleanEngine::Exact,
                    Engine::Robust => BooleanEngine::Robust,
                };
                match op {
                    BoolOp::Union => ma.union_with_engine(&mb, engine),
                    BoolOp::Difference => ma.difference_with_engine(&mb, engine),
                    BoolOp::Intersection => ma.intersection_with_engine(&mb, engine),
                }
            };
            let mut engine = if ea == Engine::Exact && eb == Engine::Exact { Engine::Exact } else { Engine::Robust };
            let mut result = run(engine);
            if result.status() != ManifoldError::NoError && engine == Engine::Exact {
                engine = Engine::Robust;
                result = run(engine);
            }
            match result.status() {
                // A robust result can itself be soup, so it keeps needing
                // the robust engine.
                ManifoldError::NoError => return Ok(Shape::Manifold(result, engine)),
                status => format!("the boolean failed ({status:?})"),
            }
        }
        (Err(status), _) | (_, Err(status)) => {
            format!("an operand isn't a closed solid ({status:?})")
        }
    };

    let polygons = a.polygon_count() + b.polygon_count();
    if polygons > BSP_FALLBACK_MAX_POLYGONS {
        return Err(format!(
            "boolean failed: {rejected}, and at {polygons} polygons the fallback \
             (limit {BSP_FALLBACK_MAX_POLYGONS}) would need too much memory - \
             make sure both shapes are closed, or lower their segment counts"
        ));
    }
    let (a, b) = (a.into_mesh(), b.into_mesh());
    Ok(Shape::Mesh(match op {
        BoolOp::Union => a.union(&b),
        BoolOp::Difference => a.difference(&b),
        BoolOp::Intersection => a.intersection(&b),
    }))
}

/// Convert a csgrs mesh into a Manifold: triangulate, then weld corners that
/// share a position (csgrs gives every polygon its own copies) so triangles
/// are connected along their edges, which Manifold needs. Tries the strict
/// import first and falls back to the one that accepts closed but
/// non-manifold "triangle soup" (e.g. T-junctions). Also decides which
/// [`Engine`] the result needs.
fn to_manifold(mesh: &Solid3) -> Result<(Manifold, Engine), ManifoldError> {
    let triangulated = mesh.triangulate();
    let mut ids: HashMap<(i64, i64, i64), u64> = HashMap::new();
    let mut vert_properties = Vec::new();
    let mut tri_verts = Vec::with_capacity(3 * triangulated.polygons.len());
    for polygon in &triangulated.polygons {
        let mut corner = [0u64; 3];
        for (slot, vertex) in corner.iter_mut().zip(&polygon.vertices) {
            let p = vertex.pos;
            *slot = *ids.entry(position_key(p.x, p.y, p.z)).or_insert_with(|| {
                vert_properties.extend([p.x, p.y, p.z]);
                (vert_properties.len() / 3 - 1) as u64
            });
        }
        // Welding can collapse a sliver triangle onto an edge; drop it.
        if corner[0] != corner[1] && corner[1] != corner[2] && corner[0] != corner[2] {
            tri_verts.extend(corner);
        }
    }
    let gl = MeshGL64 { num_prop: 3, vert_properties, tri_verts, ..Default::default() };

    let strict = Manifold::from_mesh_gl64(&gl);
    if strict.status() == ManifoldError::NoError {
        // Properly connected, but a sweep along a tight curve (say) can
        // still pass through itself.
        let engine = if strict.has_self_intersections() { Engine::Robust } else { Engine::Exact };
        return Ok((strict, engine));
    }
    let soup = Manifold::from_mesh_gl64_robust(&gl);
    match soup.status() {
        ManifoldError::NoError => Ok((soup, Engine::Robust)),
        status => Err(status),
    }
}

/// Convert a Manifold back into a csgrs mesh of triangles with flat normals.
fn from_manifold(manifold: &Manifold) -> Solid3 {
    let gl = manifold.get_mesh_gl64(-1);
    let stride = gl.num_prop as usize;
    let position = |i: u64| {
        let o = i as usize * stride;
        Point3::new(gl.vert_properties[o], gl.vert_properties[o + 1], gl.vert_properties[o + 2])
    };
    let polygons = gl
        .tri_verts
        .chunks_exact(3)
        .map(|tri| {
            let [a, b, c] = [position(tri[0]), position(tri[1]), position(tri[2])];
            let normal = (b - a).cross(&(c - a)).normalize();
            let vertices = [a, b, c].map(|p| CsgVertex::new(p, normal)).to_vec();
            CsgPolygon::new(vertices, None)
        })
        .collect::<Vec<_>>();
    Solid3::from_polygons(&polygons, None)
}

/// Hash key for merging coincident points: coordinates snapped to a 1e-6
/// grid, the same tolerance csgrs uses.
fn position_key(x: f64, y: f64, z: f64) -> (i64, i64, i64) {
    let snap = |v: f64| (v * 1e6).round() as i64;
    (snap(x), snap(y), snap(z))
}

/// A lazily-evaluated node in a 2D sketch expression tree - the flat
/// counterpart to [`Node`], turned into a [`Solid`] via `.extrude(..)`,
/// `.revolve(..)` or `.sweep(..)`.
enum SketchNode {
    Circle { radius: f64, segments: usize },
    Square { width: f64 },
    Rectangle { width: f64, length: f64 },
    Polygon { points: Vec<(f64, f64)> },
    /// A Bezier curve sampled from its control points (de Casteljau). Closed
    /// automatically into a filled region if the first and last sampled
    /// points coincide, otherwise stays an open poly-line.
    Bezier { control: Vec<(f64, f64)>, segments: usize },
    /// An open-uniform B-spline of degree `degree` through its control
    /// points - tangent-continuous (C1) across control points, unlike the
    /// polyline through the same points would be. Closed the same way as
    /// `Bezier`.
    BSpline { control: Vec<(f64, f64)>, degree: usize, segments_per_span: usize },
    Translate { child: Arc<SketchNode>, dx: f64, dy: f64 },
    Rotate { child: Arc<SketchNode>, deg: f64 },
    Scale { child: Arc<SketchNode>, sx: f64, sy: f64 },
    Union { a: Arc<SketchNode>, b: Arc<SketchNode> },
    Difference { a: Arc<SketchNode>, b: Arc<SketchNode> },
    Intersection { a: Arc<SketchNode>, b: Arc<SketchNode> },
}

/// Sample an open-uniform B-spline of degree `p` via the Cox-de Boor
/// recursion, from its start to exactly its end (`control`'s last point).
///
/// This replaces csgrs 0.20's `Sketch::bspline` sampler, which has two
/// off-by-one bugs: it walks one whole knot span *past* the end of the
/// parameter domain (where every basis function is 0, so it emits a run of
/// `(0, 0)` points - a spike to the origin that makes the outline
/// self-intersect and breaks booleans/triangulation), and it emits every
/// span boundary twice.
fn sample_bspline(control: &[(f64, f64)], degree: usize, segments_per_span: usize) -> Vec<(f64, f64)> {
    let n = control.len() - 1;
    let p = degree;
    let m = n + p + 1;
    let mut knot = Vec::with_capacity(m + 1);
    for i in 0..=m {
        if i <= p {
            knot.push(0.0);
        } else if i >= m - p {
            knot.push((n - p) as f64);
        } else {
            knot.push((i - p) as f64);
        }
    }

    fn basis(i: usize, p: usize, u: f64, knot: &[f64]) -> f64 {
        if p == 0 {
            return if u >= knot[i] && u < knot[i + 1] { 1.0 } else { 0.0 };
        }
        let denom1 = knot[i + p] - knot[i];
        let denom2 = knot[i + p + 1] - knot[i + 1];
        let term1 = if denom1.abs() < 1e-9 {
            0.0
        } else {
            (u - knot[i]) / denom1 * basis(i, p - 1, u, knot)
        };
        let term2 = if denom2.abs() < 1e-9 {
            0.0
        } else {
            (knot[i + p + 1] - u) / denom2 * basis(i + 1, p - 1, u, knot)
        };
        term1 + term2
    }

    // Every sample strictly inside [0, n - p), then the exact end point: the
    // half-open degree-0 basis is 0 everywhere at u = n - p, but a clamped
    // curve's true value there is its last control point.
    let steps = (n - p) * segments_per_span;
    let mut pts: Vec<(f64, f64)> = (0..steps)
        .map(|i| {
            let u = i as f64 / segments_per_span as f64;
            control.iter().enumerate().fold((0.0, 0.0), |(x, y), (idx, &(px, py))| {
                let b = basis(idx, p, u, &knot);
                (x + b * px, y + b * py)
            })
        })
        .collect();
    pts.push(control[n]);
    pts
}

impl SketchNode {
    fn eval(&self) -> Sketch2 {
        match self {
            SketchNode::Circle { radius, segments } => {
                Sketch2::circle(*radius, *segments, None)
            }
            SketchNode::Square { width } => Sketch2::square(*width, None),
            SketchNode::Rectangle { width, length } => {
                Sketch2::rectangle(*width, *length, None)
            }
            SketchNode::Polygon { points } => {
                let points: Vec<[f64; 2]> = points.iter().map(|&(x, y)| [x, y]).collect();
                Sketch2::polygon(&points, None)
            }
            SketchNode::Bezier { control, segments } => {
                let control: Vec<[f64; 2]> = control.iter().map(|&(x, y)| [x, y]).collect();
                Sketch2::bezier(&control, *segments, None)
            }
            SketchNode::BSpline { control, degree, segments_per_span } => {
                // Always our own sampler, never csgrs's `Sketch::bspline`
                // (see `sample_bspline` for its bugs). Closed-vs-open is
                // decided from the *input*: a control loop that repeats its
                // first point is a fillable polygon, anything else an open
                // poly-line.
                if control.len() <= *degree || *segments_per_span == 0 {
                    return Sketch2::new();
                }
                let pts = sample_bspline(control, *degree, *segments_per_span);
                let geometry = if control.first() == control.last() {
                    geo_types::Geometry::Polygon(geo_types::Polygon::new(pts.into(), vec![]))
                } else {
                    geo_types::Geometry::LineString(pts.into())
                };
                Sketch2::from_geo(geo_types::GeometryCollection(vec![geometry]), None)
            }
            SketchNode::Translate { child, dx, dy } => child.eval().translate(*dx, *dy, 0.0),
            // A sketch lies flat in the XY plane, so the only rotation that
            // keeps it planar is about Z.
            SketchNode::Rotate { child, deg } => child.eval().rotate(0.0, 0.0, *deg),
            SketchNode::Scale { child, sx, sy } => child.eval().scale(*sx, *sy, 1.0),
            SketchNode::Union { a, b } => a.eval().union(&b.eval()),
            SketchNode::Difference { a, b } => a.eval().difference(&b.eval()),
            SketchNode::Intersection { a, b } => a.eval().intersection(&b.eval()),
        }
    }
}

/// Stack size for evaluating a CSG tree.
///
/// Evaluation recurses once per tree level, and the csgrs BSP fallback in
/// [`boolean`] recurses up to about once per polygon. The thread Python
/// calls us from has only 1 MB of stack on Windows. Deliberately *not*
/// huge: a deep enough recursion should fail fast, not keep going while it
/// allocates its way through all the RAM.
const EVAL_STACK_BYTES: usize = 64 << 20; // 64 MiB

/// Run `f` on a helper thread with an `EVAL_STACK_BYTES` stack and wait for
/// it. Panics are re-raised on the calling thread (PyO3 turns them into a
/// Python `PanicException`). If the OS refuses a stack that big, fall back
/// to running `f` right here.
fn with_eval_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    // Shared so the fallback can still run `f` if the spawn fails (a failed
    // spawn drops its closure without calling it).
    let task = std::sync::Mutex::new(Some(f));
    let run = || (task.lock().unwrap().take().expect("CSG task already ran"))();
    std::thread::scope(|scope| {
        let spawned = std::thread::Builder::new()
            .name("csgpy-eval".into())
            .stack_size(EVAL_STACK_BYTES)
            .spawn_scoped(scope, run);
        match spawned {
            Ok(handle) => handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
            Err(_) => run(),
        }
    })
}

/// A last-resort cap on the whole process's memory, in force only while a
/// CSG tree is being evaluated, so that a runaway computation kills the
/// Python process instead of pushing the machine into swap until it stops
/// responding.
///
/// It's an OS limit, not a Python one: hitting it makes an allocation fail,
/// which aborts the process ("memory allocation of N bytes failed") rather
/// than raising `MemoryError`. Windows enforces it through a Job Object on
/// committed memory; Linux through `RLIMIT_DATA`, which counts reserved
/// (not just used) memory; macOS doesn't enforce `RLIMIT_DATA`, so there
/// it does nothing.
mod memory_limit {
    use std::sync::Mutex;

    /// The configured cap in bytes, `None` for no cap. Unset until first
    /// used, then defaulted by [`default_limit`].
    static LIMIT: Mutex<Option<Option<u64>>> = Mutex::new(None);

    /// `CSGPY_MEMORY_LIMIT_MB` if set (`0` means no cap), otherwise half the
    /// machine's physical RAM.
    fn default_limit() -> Option<u64> {
        match std::env::var("CSGPY_MEMORY_LIMIT_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
            Some(0) => None,
            Some(mb) => Some(mb << 20),
            None => os::physical_memory().map(|bytes| bytes / 2),
        }
    }

    pub fn set(limit: Option<u64>) {
        *LIMIT.lock().unwrap() = Some(limit);
    }

    pub fn get() -> Option<u64> {
        *LIMIT.lock().unwrap().get_or_insert_with(default_limit)
    }

    /// Holds the cap in place for as long as it's alive.
    pub struct Guard(Option<os::Previous>);

    impl Guard {
        pub fn new() -> Guard {
            Guard(get().and_then(os::apply))
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            if let Some(previous) = self.0.take() {
                os::restore(previous);
            }
        }
    }

    #[cfg(windows)]
    mod os {
        use std::ffi::c_void;
        use std::sync::OnceLock;

        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        };
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        pub type Previous = ();

        pub fn physical_memory() -> Option<u64> {
            let mut status = MEMORYSTATUSEX {
                dwLength: size_of::<MEMORYSTATUSEX>() as u32,
                ..unsafe { std::mem::zeroed() }
            };
            (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.ullTotalPhys)
        }

        /// This process's job, created and joined on first use (stored as an
        /// address: raw handles aren't `Sync`). `None` if Windows refused.
        fn job() -> Option<HANDLE> {
            static JOB: OnceLock<Option<usize>> = OnceLock::new();
            JOB.get_or_init(|| unsafe {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if job.is_null() {
                    return None;
                }
                if AssignProcessToJobObject(job, GetCurrentProcess()) == 0 {
                    CloseHandle(job);
                    return None;
                }
                Some(job as usize)
            })
            .map(|job| job as HANDLE)
        }

        fn set_limit(job: HANDLE, bytes: Option<u64>) -> bool {
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            if let Some(bytes) = bytes {
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
                info.ProcessMemoryLimit = usize::try_from(bytes).unwrap_or(usize::MAX);
            }
            let ok = unsafe {
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const c_void,
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            ok != 0
        }

        pub fn apply(bytes: u64) -> Option<Previous> {
            job().filter(|&job| set_limit(job, Some(bytes))).map(|_| ())
        }

        pub fn restore((): Previous) {
            if let Some(job) = job() {
                set_limit(job, None);
            }
        }
    }

    #[cfg(unix)]
    mod os {
        pub type Previous = libc::rlim_t;

        pub fn physical_memory() -> Option<u64> {
            let (pages, page_size) =
                unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
            (pages > 0 && page_size > 0).then(|| pages as u64 * page_size as u64)
        }

        fn current() -> Option<libc::rlimit> {
            let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
            (unsafe { libc::getrlimit(libc::RLIMIT_DATA, &mut limit) } == 0).then_some(limit)
        }

        fn set_soft(soft: libc::rlim_t) -> bool {
            let Some(mut limit) = current() else { return false };
            limit.rlim_cur = soft.min(limit.rlim_max);
            unsafe { libc::setrlimit(libc::RLIMIT_DATA, &limit) == 0 }
        }

        /// Never raises an existing, tighter soft limit.
        pub fn apply(bytes: u64) -> Option<Previous> {
            let previous = current()?.rlim_cur;
            let bytes = libc::rlim_t::try_from(bytes).unwrap_or(libc::RLIM_INFINITY);
            set_soft(bytes.min(previous)).then_some(previous)
        }

        pub fn restore(previous: Previous) {
            set_soft(previous);
        }
    }

    #[cfg(not(any(windows, unix)))]
    mod os {
        pub type Previous = ();
        pub fn physical_memory() -> Option<u64> {
            None
        }
        pub fn apply(_bytes: u64) -> Option<Previous> {
            None
        }
        pub fn restore((): Previous) {}
    }
}

/// Write an already-triangulated mesh as Wavefront OBJ.
///
/// Used instead of csgrs's own `Mesh::to_obj`, which de-duplicates vertices
/// with a linear search per vertex - quadratic, so a mesh with tens of
/// thousands of vertices takes minutes. Here positions (and normals) are
/// merged through a hash map keyed on [`position_key`].
fn write_obj(mesh: &Solid3, out: &mut impl Write) -> std::io::Result<()> {
    fn index_of(
        map: &mut HashMap<(i64, i64, i64), usize>,
        list: &mut Vec<(f64, f64, f64)>,
        v: (f64, f64, f64),
    ) -> usize {
        *map.entry(position_key(v.0, v.1, v.2)).or_insert_with(|| {
            list.push(v);
            list.len() // OBJ indices are 1-based
        })
    }

    let (mut vertex_ids, mut vertices) = (HashMap::new(), Vec::new());
    let (mut normal_ids, mut normals) = (HashMap::new(), Vec::new());
    let mut faces = Vec::with_capacity(mesh.polygons.len());
    for polygon in &mesh.polygons {
        let n = polygon.plane.normal().normalize();
        let n = if n.iter().all(|c| c.is_finite()) { (n.x, n.y, n.z) } else { (0.0, 0.0, 1.0) };
        let normal = index_of(&mut normal_ids, &mut normals, n);
        let mut corners = polygon
            .vertices
            .iter()
            .map(|v| index_of(&mut vertex_ids, &mut vertices, (v.pos.x, v.pos.y, v.pos.z)));
        if let (Some(a), Some(b), Some(c)) = (corners.next(), corners.next(), corners.next()) {
            faces.push(([a, b, c], normal));
        }
    }

    writeln!(out, "# Generated by csgpy\no csgpy")?;
    for (x, y, z) in &vertices {
        writeln!(out, "v {x:.6} {y:.6} {z:.6}")?;
    }
    for (x, y, z) in &normals {
        writeln!(out, "vn {x:.6} {y:.6} {z:.6}")?;
    }
    for ([a, b, c], n) in &faces {
        writeln!(out, "f {a}//{n} {b}//{n} {c}//{n}")?;
    }
    Ok(())
}

/// A lazy 3D solid built out of `csgrs` primitives and boolean/transform
/// operations.
///
/// Nothing is meshed until it's needed: `.to_mesh()`, `.save_stl()`,
/// `.save_obj()` (and rendering, which calls `.to_mesh()`) is what triggers
/// evaluation, and the result is cached on this instance so repeat calls
/// don't redo the work.
#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
struct Solid {
    node: Arc<Node>,
    cache: Arc<OnceLock<Solid3>>,
}

impl Solid {
    fn from_node(op: Op) -> Self {
        Solid { node: Node::new(op), cache: Arc::new(OnceLock::new()) }
    }

    fn child(&self) -> Arc<Node> {
        self.node.clone()
    }

    /// Evaluate (once) and return the mesh; a failed boolean becomes a
    /// Python `RuntimeError` and isn't cached, so it's retried next call.
    ///
    /// The evaluated shape is also kept on the node, whether or not anything
    /// shares it yet: a solid you've already shown is likely to be reused
    /// (`show(part)`, then `part.translate(..)`), and that shouldn't redo
    /// its booleans.
    fn mesh(&self) -> PyResult<&Solid3> {
        if let Some(mesh) = self.cache.get() {
            return Ok(mesh);
        }
        let mesh = {
            let _limit = memory_limit::Guard::new();
            with_eval_stack(|| {
                let shape = self.node.eval()?;
                let _ = self.node.shape.set(shape.clone());
                Ok::<_, String>(shape.into_mesh())
            })
        }
        .map_err(PyRuntimeError::new_err)?;
        Ok(self.cache.get_or_init(|| mesh))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Solid {
    /// Move this solid by `(dx, dy, dz)`. Returns a new, still-lazy `Solid`.
    fn translate(&self, dx: f64, dy: f64, dz: f64) -> Solid {
        Solid::from_node(Op::Translate { child: self.child(), dx, dy, dz })
    }

    /// Rotate this solid by the given angles in degrees, about X, Y, Z in turn.
    #[pyo3(signature = (rx=0.0, ry=0.0, rz=0.0))]
    fn rotate(&self, rx: f64, ry: f64, rz: f64) -> Solid {
        Solid::from_node(Op::Rotate { child: self.child(), rx_deg: rx, ry_deg: ry, rz_deg: rz })
    }

    /// Scale this solid by `(sx, sy, sz)`, or uniformly if only `sx` is given.
    #[pyo3(signature = (sx, sy=None, sz=None))]
    fn scale(&self, sx: f64, sy: Option<f64>, sz: Option<f64>) -> Solid {
        Solid::from_node(Op::Scale {
            child: self.child(),
            sx,
            sy: sy.unwrap_or(sx),
            sz: sz.unwrap_or(sx),
        })
    }

    /// Boolean union with `other`. Same as `self + other`.
    fn union(&self, other: &Solid) -> Solid {
        Solid::from_node(Op::Union { a: self.child(), b: other.child() })
    }

    /// Boolean subtraction of `other` from `self`. Same as `self - other`.
    fn subtract(&self, other: &Solid) -> Solid {
        Solid::from_node(Op::Difference { a: self.child(), b: other.child() })
    }

    /// Boolean intersection with `other`. Same as `self & other`.
    fn intersect(&self, other: &Solid) -> Solid {
        Solid::from_node(Op::Intersection { a: self.child(), b: other.child() })
    }

    fn __add__(&self, other: &Solid) -> Solid {
        self.union(other)
    }

    fn __or__(&self, other: &Solid) -> Solid {
        self.union(other)
    }

    fn __sub__(&self, other: &Solid) -> Solid {
        self.subtract(other)
    }

    fn __and__(&self, other: &Solid) -> Solid {
        self.intersect(other)
    }

    /// Force evaluation of the lazy tree (cached after the first call) and
    /// return `(vertices, triangles)`: a list of `(x, y, z)` points and a
    /// list of `(i, j, k)` index triples into that list.
    fn to_mesh(&self) -> PyResult<(Vec<(f64, f64, f64)>, Vec<(u32, u32, u32)>)> {
        let triangulated = self.mesh()?.triangulate();

        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for polygon in &triangulated.polygons {
            // `triangulate()` guarantees every polygon has exactly 3 vertices.
            let base = vertices.len() as u32;
            for vertex in &polygon.vertices {
                let p = vertex.pos;
                vertices.push((p.x, p.y, p.z));
            }
            triangles.push((base, base + 1, base + 2));
        }
        Ok((vertices, triangles))
    }

    /// Write this solid to an STL file - the standard format for 3D printing
    /// (slicers) and CAD. Binary by default (compact, fast to load); pass
    /// `ascii=True` for a human-readable text file instead.
    #[pyo3(signature = (path, ascii=false))]
    fn save_stl(&self, path: PathBuf, ascii: bool) -> PyResult<()> {
        let mesh = self.mesh()?;
        if ascii {
            std::fs::write(path, mesh.to_stl_ascii("csgpy"))?;
        } else {
            std::fs::write(path, mesh.to_stl_binary("csgpy")?)?;
        }
        Ok(())
    }

    /// Write this solid to a Wavefront OBJ file - the most widely supported
    /// general 3D format (Blender, game engines, web viewers, CAD tools).
    ///
    /// Corners at the same position are written once and shared between
    /// triangles (unlike STL, which repeats them per triangle); each
    /// triangle gets a flat per-face normal.
    fn save_obj(&self, path: PathBuf) -> PyResult<()> {
        let mut out = BufWriter::new(File::create(path)?);
        write_obj(&self.mesh()?.triangulate(), &mut out)?;
        out.flush()?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let evaluated = self.cache.get().is_some();
        format!("Solid(evaluated={evaluated})")
    }
}

/// A lazily-evaluated 2D shape in the XY plane, built out of `csgpy`
/// primitives (`circle(..)`, `square(..)`, `rectangle(..)`, `polygon(..)`)
/// and boolean/transform operations - the 2D counterpart to [`Solid`].
///
/// Turn it into a [`Solid`] with `.extrude(..)`, `.extrude_vector(..)`,
/// `.revolve(..)` or `.sweep(..)`.
#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
struct Sketch {
    node: Arc<SketchNode>,
}

impl Sketch {
    fn from_node(node: SketchNode) -> Self {
        Sketch { node: Arc::new(node) }
    }

    fn child(&self) -> Arc<SketchNode> {
        self.node.clone()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Sketch {
    /// Move this sketch by `(dx, dy)`. Returns a new, still-lazy `Sketch`.
    fn translate(&self, dx: f64, dy: f64) -> Sketch {
        Sketch::from_node(SketchNode::Translate { child: self.child(), dx, dy })
    }

    /// Rotate this sketch by `deg` degrees about the origin (a sketch is
    /// planar, so - unlike `Solid.rotate` - there's only one axis to turn
    /// around).
    fn rotate(&self, deg: f64) -> Sketch {
        Sketch::from_node(SketchNode::Rotate { child: self.child(), deg })
    }

    /// Scale this sketch by `(sx, sy)`, or uniformly if only `sx` is given.
    #[pyo3(signature = (sx, sy=None))]
    fn scale(&self, sx: f64, sy: Option<f64>) -> Sketch {
        Sketch::from_node(SketchNode::Scale { child: self.child(), sx, sy: sy.unwrap_or(sx) })
    }

    /// Boolean union with `other`. Same as `self + other`.
    fn union(&self, other: &Sketch) -> Sketch {
        Sketch::from_node(SketchNode::Union { a: self.child(), b: other.child() })
    }

    /// Boolean subtraction of `other` from `self`. Same as `self - other`.
    fn subtract(&self, other: &Sketch) -> Sketch {
        Sketch::from_node(SketchNode::Difference { a: self.child(), b: other.child() })
    }

    /// Boolean intersection with `other`. Same as `self & other`.
    fn intersect(&self, other: &Sketch) -> Sketch {
        Sketch::from_node(SketchNode::Intersection { a: self.child(), b: other.child() })
    }

    fn __add__(&self, other: &Sketch) -> Sketch {
        self.union(other)
    }

    fn __or__(&self, other: &Sketch) -> Sketch {
        self.union(other)
    }

    fn __sub__(&self, other: &Sketch) -> Sketch {
        self.subtract(other)
    }

    fn __and__(&self, other: &Sketch) -> Sketch {
        self.intersect(other)
    }

    /// Linearly extrude this sketch `height` units up the Z axis into a
    /// [`Solid`]. Shorthand for `extrude_vector(0, 0, height)`.
    fn extrude(&self, height: f64) -> Solid {
        self.extrude_vector(0.0, 0.0, height)
    }

    /// Linearly extrude this sketch along an arbitrary `(dx, dy, dz)`
    /// direction vector into a [`Solid`] - e.g. a non-zero `dx`/`dy` gives a
    /// slanted (sheared) extrusion instead of a straight-up one.
    fn extrude_vector(&self, dx: f64, dy: f64, dz: f64) -> Solid {
        Solid::from_node(Op::ExtrudeVector { sketch: self.child(), dx, dy, dz })
    }

    /// Revolve this sketch around the Y axis into a surface of revolution
    /// (a lathed [`Solid`]) - e.g. a rectangle offset from the axis makes a
    /// ring/tube, a right-triangle profile makes a cone.
    ///
    /// `angle_degs` is the sweep angle (360 for a full solid of revolution,
    /// less for a wedge with flat end caps); `segments` is the number of
    /// angular subdivisions and must be at least 2.
    #[pyo3(signature = (angle_degs=360.0, segments=32))]
    fn revolve(&self, angle_degs: f64, segments: usize) -> PyResult<Solid> {
        if segments < 2 {
            return Err(PyValueError::new_err("revolve() needs at least 2 segments"));
        }
        Ok(Solid::from_node(Op::Revolve {
            sketch: self.child(),
            angle_deg: angle_degs,
            segments,
        }))
    }

    /// Sweep (extrude-along-a-path): duplicate this sketch at every point of
    /// `path` (a list of `(x, y, z)` points), aiming its +Z at the local
    /// path tangent, and stitch the copies into a [`Solid`] - e.g. a circle
    /// swept along points on an arc, or along a helix for a screw-like
    /// shape. If `path`'s first and last points coincide it's treated as a
    /// closed loop and no end caps are added.
    fn sweep(&self, path: Vec<(f64, f64, f64)>) -> PyResult<Solid> {
        if path.len() < 2 {
            return Err(PyValueError::new_err("sweep() needs a path of at least 2 points"));
        }
        Ok(Solid::from_node(Op::Sweep { sketch: self.child(), path }))
    }

    fn __repr__(&self) -> String {
        "Sketch()".to_string()
    }
}

/// An axis-aligned box (cuboid) of the given width (X), length (Y) and
/// height (Z), centered on the origin's corner at `(0, 0, 0)`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "box", signature = (width, length, height))]
fn box_(width: f64, length: f64, height: f64) -> Solid {
    Solid::from_node(Op::Cuboid { width, length, height })
}

/// A cylinder of the given radius and height, standing on the XY plane
/// with its base centered at `(0, 0, 0)`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (radius, height, segments=32))]
fn cylinder(radius: f64, height: f64, segments: usize) -> Solid {
    Solid::from_node(Op::Cylinder { radius, height, segments })
}

/// A sphere of the given radius, centered at `(0, 0, 0)`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (radius, segments=32, stacks=16))]
fn sphere(radius: f64, segments: usize, stacks: usize) -> Solid {
    Solid::from_node(Op::Sphere { radius, segments, stacks })
}

/// A cone of the given base radius and height, standing on the XY plane
/// with its base centered at `(0, 0, 0)`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (radius, height, segments=32))]
fn cone(radius: f64, height: f64, segments: usize) -> Solid {
    Solid::from_node(Op::Cone { radius, height, segments })
}

/// A 2D circle of the given radius in the XY plane, centered on the
/// origin.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (radius, segments=32))]
fn circle(radius: f64, segments: usize) -> Sketch {
    Sketch::from_node(SketchNode::Circle { radius, segments })
}

/// A 2D square of the given width, in the XY plane, with one corner at
/// `(0, 0)`.
#[gen_stub_pyfunction]
#[pyfunction]
fn square(width: f64) -> Sketch {
    Sketch::from_node(SketchNode::Square { width })
}

/// A 2D rectangle of the given width (X) and length (Y), with one corner
/// at `(0, 0)`.
#[gen_stub_pyfunction]
#[pyfunction]
fn rectangle(width: f64, length: f64) -> Sketch {
    Sketch::from_node(SketchNode::Rectangle { width, length })
}

/// A 2D polygon in the XY plane from an ordered list of `(x, y)` points
/// (closed automatically if the first and last points don't already
/// match).
#[gen_stub_pyfunction]
#[pyfunction]
fn polygon(points: Vec<(f64, f64)>) -> Sketch {
    Sketch::from_node(SketchNode::Polygon { points })
}

/// A Bezier curve in the XY plane, sampled from its `control` points. If the
/// sampled curve comes back around to its start (e.g. the last control
/// point equals the first) it's a closed, fillable profile - otherwise it's
/// an open poly-line (useful as a `sweep()` path, not as an `extrude()`
/// base).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (control, segments=32))]
fn bezier(control: Vec<(f64, f64)>, segments: usize) -> Sketch {
    Sketch::from_node(SketchNode::Bezier { control, segments })
}

/// An open-uniform B-spline of the given `degree` (3 = cubic) through
/// `control` points, in the XY plane - a *soft*, tangent-continuous curve
/// (unlike `polygon()`, which is a sharp-cornered polyline through the same
/// points). Closed the same way as `bezier()`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (control, degree=3, segments_per_span=8))]
fn bspline(control: Vec<(f64, f64)>, degree: usize, segments_per_span: usize) -> Sketch {
    Sketch::from_node(SketchNode::BSpline { control, degree, segments_per_span })
}

/// Boolean union of `a` and `b`. Same as `a + b`.
#[gen_stub_pyfunction]
#[pyfunction]
fn union(a: &Solid, b: &Solid) -> Solid {
    a.union(b)
}

/// Boolean subtraction of `b` from `a`. Same as `a - b`.
#[gen_stub_pyfunction]
#[pyfunction]
fn subtract(a: &Solid, b: &Solid) -> Solid {
    a.subtract(b)
}

/// Boolean intersection of `a` and `b`. Same as `a & b`.
#[gen_stub_pyfunction]
#[pyfunction]
fn intersect(a: &Solid, b: &Solid) -> Solid {
    a.intersect(b)
}

/// Cap how much memory this Python process may use while csgpy evaluates a
/// solid, in megabytes (`None` for no cap). A computation that runs past it
/// ends the process instead of making the whole machine unresponsive.
///
/// Defaults to half the machine's RAM, or the `CSGPY_MEMORY_LIMIT_MB`
/// environment variable if set (`0` meaning no cap). Enforced on Windows and
/// Linux; a no-op on macOS.
#[gen_stub_pyfunction]
#[pyfunction]
fn set_memory_limit(megabytes: Option<u64>) {
    memory_limit::set(megabytes.map(|mb| mb.saturating_mul(1 << 20)));
}

/// The current cap from `set_memory_limit`, in megabytes (`None` if uncapped).
#[gen_stub_pyfunction]
#[pyfunction]
fn get_memory_limit() -> Option<u64> {
    memory_limit::get().map(|bytes| bytes >> 20)
}

/// The Python module. Its name must match `[lib].name` in Cargo.toml and
/// `[project].name` in pyproject.toml.
#[pymodule]
fn csgpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solid>()?;
    m.add_class::<Sketch>()?;
    m.add_function(wrap_pyfunction!(box_, m)?)?;
    m.add_function(wrap_pyfunction!(cylinder, m)?)?;
    m.add_function(wrap_pyfunction!(sphere, m)?)?;
    m.add_function(wrap_pyfunction!(cone, m)?)?;
    m.add_function(wrap_pyfunction!(circle, m)?)?;
    m.add_function(wrap_pyfunction!(square, m)?)?;
    m.add_function(wrap_pyfunction!(rectangle, m)?)?;
    m.add_function(wrap_pyfunction!(polygon, m)?)?;
    m.add_function(wrap_pyfunction!(bezier, m)?)?;
    m.add_function(wrap_pyfunction!(bspline, m)?)?;
    m.add_function(wrap_pyfunction!(union, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(intersect, m)?)?;
    m.add_function(wrap_pyfunction!(set_memory_limit, m)?)?;
    m.add_function(wrap_pyfunction!(get_memory_limit, m)?)?;
    Ok(())
}

// Collects everything tagged with #[gen_stub_*] above into a `stub_info()`
// function that src/bin/stub_gen.rs calls to write csgpy.pyi.
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);

