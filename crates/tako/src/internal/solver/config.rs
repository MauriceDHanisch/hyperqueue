use std::time::Duration;

/// Relative MIP optimality gap: the solver accepts a solution once it is
/// provably within this fraction of the true optimum, instead of always
/// proving exact optimality. Task resource requests are themselves estimates
/// (cpu/memory bucketing), so demanding an exact optimum is false precision;
/// a 10% gap trades a small, usually much smaller in practice, placement
/// suboptimality for a solve that reliably finishes in well under a second on
/// realistic instances instead of stalling the single-threaded server.
pub(crate) fn mip_rel_gap() -> f64 {
    // Unit tests assert exact placement counts/priority behavior on small,
    // fast-solving instances -- a nonzero default gap there would trade
    // correctness-test fidelity for a speedup these tiny instances don't
    // need. A test that wants to exercise gap-tuned behavior specifically
    // uses with_test_solver_config below (thread-local, not process-global
    // env vars, so it can't race with unrelated tests running concurrently
    // on other threads).
    #[cfg(test)]
    if let Some(v) = TEST_REL_GAP_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    #[cfg(test)]
    let default = 0.0;
    #[cfg(not(test))]
    let default = 0.10;

    get_f64_from_env("HQ_SCHEDULER_MIP_REL_GAP").unwrap_or(default)
}

/// Hard wall-clock cap on a single scheduling solve. The solver is otherwise
/// unbounded and can run for minutes to hours on workloads with many distinct
/// resource shapes, blocking the single-threaded server (no heartbeats, no
/// RPCs, no other scheduling) for the entire duration. 5s clears the steep
/// part of the incumbent-quality cliff observed on realistic and
/// harder-than-realistic synthetic instances while bounding the worst case.
pub(crate) fn mip_time_limit() -> Duration {
    // See mip_rel_gap: unit tests need exact, unhurried solves on tiny
    // instances, not a production-scale wall-clock bound.
    #[cfg(test)]
    if let Some(v) = TEST_TIME_LIMIT_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    #[cfg(test)]
    let default = Duration::from_secs(60);
    #[cfg(not(test))]
    let default = Duration::from_secs(5);

    get_duration_from_env("HQ_SCHEDULER_MIP_TIME_LIMIT_MS").unwrap_or(default)
}

/// Number of threads HiGHS may use internally for one solve. Both solve()
/// and solve_bounded() run MILPs this code's own docs call tiny -- no
/// solve-quality benefit to internal parallelism here. Left unset, HiGHS's
/// own heuristic auto-detects all visible cores and can burst-spawn far
/// more native OS threads than a solve this size needs; on a host with a
/// low max-user-processes ulimit that burst can exceed the limit, throwing
/// an uncaught C++ exception across the FFI boundary that calls
/// std::terminate() and kills the whole hq server process outright.
///
/// Vista-specific (see highs.rs's call sites, gated to aarch64): observed only on
/// that site's login node (tight `ulimit -u`), not on any of this fork's x86 sites
/// -- gated by CPU architecture since Vista is currently this fork's only aarch64
/// deployment. Unset, this now scales with detect_safe_thread_count() (hardware
/// cores clamped by the actual, currently observed ulimit headroom) rather than a
/// blind guess -- a fixed default of 1 was too conservative for solve quality at
/// scale, but the site's own hand-picked override of 32 (see sites.py) still
/// wasn't safe under every real load, motivating an environment-aware value
/// instead of another fixed guess. HQ_SCHEDULER_MIP_THREADS still overrides both.
#[cfg(target_arch = "aarch64")]
pub(crate) fn mip_threads() -> i32 {
    static DEFAULT: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    get_i32_from_env("HQ_SCHEDULER_MIP_THREADS")
        .unwrap_or_else(|| *DEFAULT.get_or_init(detect_safe_thread_count))
}

/// Picks a thread count that scales with available hardware while staying under this
/// process's actual, currently observed `ulimit -u` (RLIMIT_NPROC) headroom -- computed once
/// per process (via mip_threads' OnceLock) rather than on every solve, since a full process
/// scan on every scheduling tick would add real latency at scale for a value that only needs
/// to be a reasonable snapshot, not perfectly live.
///
/// RLIMIT_NPROC is a whole-*user* limit (every thread of every process the user owns
/// system-wide counts against it, not just this process's own threads), so both the limit and
/// the current usage have to be read at that same scope -- a process-local thread count would
/// silently ignore everything else already running under the account (other jobs, ssh
/// sessions, the shell itself) that shares the same ceiling.
#[cfg(target_arch = "aarch64")]
fn detect_safe_thread_count() -> i32 {
    let cores = std::thread::available_parallelism().map(|n| n.get() as i64).unwrap_or(1);

    let Ok((soft_limit, _hard_limit)) =
        nix::sys::resource::getrlimit(nix::sys::resource::Resource::RLIMIT_NPROC)
    else {
        return cores.max(1) as i32;
    };
    if soft_limit == nix::sys::resource::RLIM_INFINITY {
        // No real ceiling to protect against (the common case off Vista) -- use every core.
        return cores.max(1) as i32;
    }

    let current_threads = count_current_user_threads().unwrap_or(0);
    // Headroom left under the ulimit, minus a safety margin for threads this same process
    // still needs to spawn after this point (tokio's own worker threads, jemalloc's
    // background thread, new connections, concurrent solves) and for the rest of the
    // account's usage to fluctuate before this value is reused on a later solve.
    const SAFETY_MARGIN: i64 = 16;
    let headroom = soft_limit as i64 - current_threads - SAFETY_MARGIN;

    headroom.clamp(1, cores) as i32
}

/// Total thread count currently charged against this user's RLIMIT_NPROC: every thread of
/// every process this user owns system-wide (RLIMIT_NPROC counts threads, not just processes,
/// on Linux), found by scanning /proc rather than just this process's own thread count.
#[cfg(target_arch = "aarch64")]
fn count_current_user_threads() -> Option<i64> {
    let my_uid = nix::unistd::getuid();
    let mut total = 0i64;
    for entry in std::fs::read_dir("/proc").ok()?.flatten() {
        if !entry.file_name().to_str().is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit())) {
            continue; // not a /proc/<pid> entry
        }
        // A process can exit mid-scan, or its status file can be transiently unreadable --
        // neither is fatal to an approximate, best-effort count, so just skip it.
        let Ok(status) = std::fs::read_to_string(entry.path().join("status")) else {
            continue;
        };
        let mut is_mine = false;
        let mut threads = 0i64;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("Uid:") {
                is_mine = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u32>().ok())
                    .is_some_and(|uid| uid == my_uid.as_raw());
            } else if let Some(rest) = line.strip_prefix("Threads:") {
                threads = rest.trim().parse().unwrap_or(0);
            }
        }
        if is_mine {
            total += threads;
        }
    }
    Some(total)
}

fn get_f64_from_env(key: &str) -> Option<f64> {
    std::env::var(key).ok().and_then(|value| value.parse::<f64>().ok())
}

#[cfg(target_arch = "aarch64")]
fn get_i32_from_env(key: &str) -> Option<i32> {
    std::env::var(key).ok().and_then(|value| value.parse::<i32>().ok())
}

fn get_duration_from_env(key: &str) -> Option<Duration> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
}

#[cfg(test)]
thread_local! {
    static TEST_REL_GAP_OVERRIDE: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
    static TEST_TIME_LIMIT_OVERRIDE: std::cell::Cell<Option<Duration>> = const { std::cell::Cell::new(None) };
}

/// Runs `f` with a scheduler solver config override in effect, for tests
/// that need to exercise the production-tuned (or otherwise non-default)
/// solve_bounded() behavior. Thread-local, not a process-global env var: the
/// Rust test harness runs each #[test] to completion on its own OS thread,
/// so this cannot race with unrelated tests running concurrently on other
/// threads the way a process-global env var would.
#[cfg(test)]
pub(crate) fn with_test_solver_config<R>(rel_gap: f64, time_limit: Duration, f: impl FnOnce() -> R) -> R {
    TEST_REL_GAP_OVERRIDE.with(|c| c.set(Some(rel_gap)));
    TEST_TIME_LIMIT_OVERRIDE.with(|c| c.set(Some(time_limit)));
    let result = f();
    TEST_REL_GAP_OVERRIDE.with(|c| c.set(None));
    TEST_TIME_LIMIT_OVERRIDE.with(|c| c.set(None));
    result
}

#[cfg(all(test, target_arch = "aarch64"))]
mod thread_detect_tests {
    use super::*;

    #[test]
    fn count_current_user_threads_sees_at_least_this_process() {
        // This process itself has at least one thread, and it's owned by us, so the scan
        // should never come back empty on any real system.
        let n = count_current_user_threads().expect("proc scan should succeed under test");
        assert!(n >= 1, "expected at least 1 thread, got {n}");
    }

    #[test]
    fn detect_safe_thread_count_stays_within_hardware_bounds() {
        let cores = std::thread::available_parallelism().unwrap().get() as i32;
        let n = detect_safe_thread_count();
        assert!((1..=cores).contains(&n), "expected 1..={cores}, got {n}");
    }
}
