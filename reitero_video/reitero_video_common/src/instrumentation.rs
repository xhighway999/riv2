use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Simple thread-local instrumentation helper.
// API:
//   Instrument::start_measure("name");
//   Instrument::stop_measure("name");
//   Instrument::dump_to_stdout() or dump_to_dir("penginstrument")

pub struct Measurement {
    pub count: u64,
    pub total_ns: u128,
}

impl Measurement {
    fn new() -> Self {
        Measurement { count: 0, total_ns: 0 }
    }
}

thread_local! {
    static INST: RefCell<InstrumentInner> = RefCell::new(InstrumentInner::new());
}

pub struct Instrument;

struct InstrumentInner {
    // name -> (start_instant stack) to allow nested starts
    starts: HashMap<String, Vec<Instant>>,
    // name -> Measurement
    measurements: HashMap<String, Measurement>,
}

impl InstrumentInner {
    fn new() -> Self {
        InstrumentInner {
            starts: HashMap::new(),
            measurements: HashMap::new(),
        }
    }
}

impl Instrument {
    pub fn start_measure(name: &str) {
        let key = name.to_string();
        INST.with(|i| {
            let mut inner = i.borrow_mut();
            inner.starts.entry(key).or_default().push(Instant::now());
        });
    }

    pub fn stop_measure(name: &str) {
        let key = name.to_string();
        INST.with(|i| {
            let mut inner = i.borrow_mut();
            // remove the stack to avoid holding multiple mutable borrows into the map
            if let Some(mut stack) = inner.starts.remove(&key) {
                if let Some(start) = stack.pop() {
                    let dur = start.elapsed();
                    let m = inner.measurements.entry(key.clone()).or_insert_with(Measurement::new);
                    m.count += 1;
                    m.total_ns += dur.as_nanos();
                }
                if !stack.is_empty() {
                    // put it back
                    inner.starts.insert(key, stack);
                }
            }
        });
    }

    pub fn dump_to_stdout() {
        INST.with(|i| {
            let inner = i.borrow();
            println!("=== Instrumentation Dump ===");
            for (k, v) in inner.measurements.iter() {
                let avg_ns = if v.count > 0 { v.total_ns / (v.count as u128) } else { 0 };
                println!("{:<30} count {:>6} total_ms {:>8.3} avg_ms {:>6.3}", k, v.count, (v.total_ns as f64) / 1e6, (avg_ns as f64) / 1e6);
            }
        });
    }

    pub fn dump_to_dir(dir: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        std::fs::create_dir_all(dir)?;
        INST.with(|i| {
            let inner = i.borrow();
            let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            let path = format!("{}/instrument_{}.txt", dir, secs);
            let mut f = File::create(&path)?;
            writeln!(f, "=== Instrumentation Dump ===")?;
            for (k, v) in inner.measurements.iter() {
                let avg_ns = if v.count > 0 { v.total_ns / (v.count as u128) } else { 0 };
                writeln!(f, "{:<30} count {:>6} total_ms {:>8.3} avg_ms {:>6.3}", k, v.count, (v.total_ns as f64) / 1e6, (avg_ns as f64) / 1e6)?;
            }
            Ok(())
        })
    }
}
