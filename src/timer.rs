use chrono::{DateTime, Local};
use std::ops::Sub;

pub struct Timer {
    name: &'static str,
    start: DateTime<Local>,
}

impl Timer {
    pub fn start(name: &'static str) -> Timer {
        let timer = Timer {
            name,
            start: Local::now(),
        };
        println!("{} Started {}", timer.start, name);
        timer
    }

    pub fn end(self) {
        let end = Local::now();
        println!(
            "{} Finished {}, took: {}",
            end,
            self.name,
            end.sub(self.start)
        );
    }
}

/// Usage:
/// ```
/// time_it!("some timer name",
///   let x = 10;
///   println!("{}", x)
/// )
/// ```
///
#[macro_export]
macro_rules! time_it {
    ($context:literal, $($tt:tt)+) => {
        let timer = crate::timer::Timer::start($context);
        $(
            $tt
        )+
        timer.end();
    }
}
