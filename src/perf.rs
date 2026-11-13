use std::collections::VecDeque;

/// Rolling window of the most recent frame durations, in milliseconds.
pub struct FrameTimes {
    samples: VecDeque<f32>,
    capacity: usize,
}

impl FrameTimes {
    /// Creates an empty window that keeps the last `capacity` frames.
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Records a frame duration, dropping the oldest one when the window is full.
    pub fn push(&mut self, ms: f32) {
        if self.capacity == 0 {
            return;
        }
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    /// Forgets every recorded frame, so the next measurement starts clean.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Returns the mean frame duration, or `None` before the first frame.
    pub fn average(&self) -> Option<f32> {
        if self.samples.is_empty() {
            return None;
        }

        Some(self.samples.iter().sum::<f32>() / self.samples.len() as f32)
    }

    /// Returns the longest frame duration in the window, or `None` before the first frame.
    pub fn max(&self) -> Option<f32> {
        self.samples.iter().copied().reduce(f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_none() {
        let times = FrameTimes::new(4);

        assert_eq!(times.average(), None);
        assert_eq!(times.max(), None);
    }

    #[test]
    fn average_and_max() {
        let mut times = FrameTimes::new(4);
        for ms in [10.0, 30.0, 20.0, 40.0] {
            times.push(ms);
        }

        assert_eq!(times.average(), Some(25.0));
        assert_eq!(times.max(), Some(40.0));
    }

    #[test]
    fn full_window_drops_oldest() {
        let mut times = FrameTimes::new(3);
        for ms in [100.0, 1.0, 2.0, 3.0] {
            times.push(ms);
        }

        // The 100 ms frame was pushed out.
        assert_eq!(times.average(), Some(2.0));
        assert_eq!(times.max(), Some(3.0));
    }

    #[test]
    fn clear_empties_the_window() {
        let mut times = FrameTimes::new(3);
        times.push(17_528.0);
        times.clear();

        assert_eq!(times.average(), None);

        times.push(4.0);
        assert_eq!(times.max(), Some(4.0));
    }

    #[test]
    fn zero_capacity_keeps_nothing() {
        let mut times = FrameTimes::new(0);
        times.push(5.0);

        assert_eq!(times.average(), None);
    }
}
