use std::cmp::Ordering;
use std::error::Error;
use std::fmt::Debug;
use rand::{Rng, thread_rng};
use rayon::prelude::*;

fn insert_within<T: Copy>(data: &mut [T], a: usize, b: usize) {
    if a == b {return};
    let (pluck_from, insert_at) = if a > b {
        (a, b)
    } else {
        (b, a)
    };

    let plucked = data[pluck_from];
    data.copy_within(insert_at..pluck_from, insert_at+1);
    data[insert_at] = plucked;
}

fn mutate_one<T: Copy>(data: &mut [T], mutation_chance: f64) {
    let mut rng = thread_rng();
    for _ in 0..data.len() {
        if rng.gen_bool(mutation_chance) {
            let a = rng.gen_range(0..data.len());
            let b = rng.gen_range(0..data.len());
            insert_within(data, a, b);
        }
    }
}

fn mutate<T: Sync + Send + Copy>(population: &mut [Vec<T>], mutation_chance: f64) {
    population.par_iter_mut()
        .for_each(|list| {
            mutate_one(list, mutation_chance);
        });
}

fn score_one_from_start<T: Ord>(data: &[T]) -> usize {
    let mut score = 1;
    let mut last = &data[0];
    for current in data.iter().skip(1) {
        match current.cmp(last) {
            Ordering::Less => break,
            Ordering::Equal => score += 1,
            Ordering::Greater => score += 1,
        }
        last = current;
    }
    score
}

fn score_one_everywhere<T: Ord>(data: &[T]) -> usize {
    data.iter()
        .skip(1)
        .fold((1, &data[0]), |(current_score, last), current| {
            let this_score = match current.cmp(last) {
                Ordering::Less => 0,
                Ordering::Equal => 1,
                Ordering::Greater => 1,
            };
            (current_score + this_score, current)
        })
        .0
}

fn score_one<T: Ord>(data: &[T]) -> f64 {
    score_one_everywhere(data) as f64 * 0.45 + score_one_from_start(data) as f64 * 0.55
}

fn conform_to_best<T: Ord + Copy + Sync + Send>(population: &mut [Vec<T>], last_best_score: f64, last_best_i: usize, score_one_fn: fn(&[T]) -> f64) -> (f64, usize) {
    let mut best_score = 0.0;
    let mut best_i = 0;
    let mut best = &population[last_best_i];

    let scores : Vec<_> = population.par_iter()
        .map(|list| score_one_fn(list))
        .collect();
    for ((i, list), score) in population.iter().enumerate().zip(scores) {
        if score > best_score {
            best_score = score;
            best_i = i;
            best = list;
        }
    }

    if last_best_score > best_score {
        best_score = last_best_score;
        best_i = last_best_i;
    }

    let best = best.clone();
    for list in population {
        list.copy_from_slice(&best);
    }

    (best_score, best_i)
}


pub trait Optimizer {

    /// Gets called once before each training session
    fn start(&mut self, data_length: usize);


    /// Gets called before each epoch
    fn get_mutation_chance(&mut self, last_done_fraction: f64) -> f64;

}


#[derive(Clone, Debug)]
pub struct SimpleOptimizer {
    ten_initial_mutation_chance: f64, // Initial mutation chance for a list of 10 elements
    exp_rate: f64,
    initial_mutation_chance: f64
}

impl Default for SimpleOptimizer {
    fn default() -> Self {
        Self {
            ten_initial_mutation_chance: 0.22,
            exp_rate: -1.14,
            initial_mutation_chance: 0.0,
        }
    }
}

impl SimpleOptimizer {

    pub fn new(ten_initial_mutation_chance: f64, decay: f64) -> Self {
        Self {
            ten_initial_mutation_chance,
            exp_rate: -decay,
            initial_mutation_chance: 0.0,
        }
    }

}

impl Optimizer for SimpleOptimizer {
    fn start(&mut self, data_length: usize) {
        self.initial_mutation_chance = self.ten_initial_mutation_chance * 10.0 / data_length as f64;
    }

    fn get_mutation_chance(&mut self, last_done_fraction: f64) -> f64 {
        self.initial_mutation_chance * (self.exp_rate * last_done_fraction).exp()
    }
}


#[derive(Clone, Debug)]
pub struct ComplexOptimizer {
    simple: SimpleOptimizer,
    last_done_fraction: f64,
    stuck_mutation_change: f64,
    stuck_decrease_by: f64,
    omega_factor: f64,
}

impl ComplexOptimizer {

    pub fn new(simple_optimizer: SimpleOptimizer, stuck_decrease_by: f64, omega_factor: f64) -> Self {
        Self {
            simple: simple_optimizer,
            last_done_fraction: 0.0,
            stuck_mutation_change: 0.0,
            omega_factor,
            stuck_decrease_by,
        }
    }

}

impl Optimizer for ComplexOptimizer {
    fn start(&mut self, data_length: usize) {
        self.simple.start(data_length);
        self.last_done_fraction = f64::NAN;
    }

    fn get_mutation_chance(&mut self, last_done_fraction: f64) -> f64 {
        if last_done_fraction == self.last_done_fraction {
            self.stuck_mutation_change -= self.stuck_decrease_by;
        } else {
            self.stuck_mutation_change /= self.omega_factor;
        }
        self.last_done_fraction = last_done_fraction;
        let mut mutation_chance = self.simple.get_mutation_chance(last_done_fraction);
        mutation_chance += mutation_chance * self.stuck_mutation_change.clamp(-1.0, 1.0) / 1.1;
        mutation_chance
    }
}


#[derive(Clone, Debug)]
pub struct BogoConfig<O: Optimizer, T: Send + Sync + Debug + Copy + Clone + Ord + PartialEq> {
    pub give_up_count: usize,
    pub verbose: bool,
    pub population_size: usize,
    pub optimizer: O,
    pub scoring_function: fn(&[T]) -> f64
}

impl<T: Send + Sync + Debug + Copy + Clone + Ord + PartialEq> Default for BogoConfig<SimpleOptimizer, fn(&[T]) -> f64> {
    fn default() -> Self {
        Self {
            give_up_count: usize::MAX,
            verbose: true,
            population_size: 64,
            optimizer: SimpleOptimizer::default(),
            scoring_function: score_one
        }
    }
}

#[derive(Debug)]
pub enum SortingResultKind<T: Ord + Copy + Sync + Send + Debug + Clone + PartialEq> {
    SortingWasOkKindResult(Vec<T>),
    SortingWasIncompleteKindResult(Vec<T>),
    SortingWasErroneousKindResult(Box<dyn Error>)
}

impl<T: Ord + Copy + Sync + Send + Debug + Clone + PartialEq> SortingResultKind<T> {

    pub fn unwrap_sorting_result_kind(self) -> Option<Vec<T>> {
        match self {
            SortingResultKind::SortingWasOkKindResult(v) => Some(v),
            SortingResultKind::SortingWasIncompleteKindResult(_) => None,
            SortingResultKind::SortingWasErroneousKindResult(_) => None,
        }
    }

    pub fn unwrap_sorting_result_kind_with_incomplete(self) -> Option<Vec<T>> {
        match self {
            SortingResultKind::SortingWasOkKindResult(v) => Some(v),
            SortingResultKind::SortingWasIncompleteKindResult(v) => Some(v),
            SortingResultKind::SortingWasErroneousKindResult(_) => None,
        }
    }

}

impl<T: Ord + Copy + Sync + Send + Debug + Clone + PartialEq> PartialEq for &SortingResultKind<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SortingResultKind::SortingWasOkKindResult(a), SortingResultKind::SortingWasOkKindResult(b)) => a == b,
            (SortingResultKind::SortingWasIncompleteKindResult(a), SortingResultKind::SortingWasIncompleteKindResult(b)) => a == b,
            _ => false
        }
    }
}


pub struct SortingResult<'a, T: Ord + Copy + Sync + Send + Debug + Clone + PartialEq, C: ?Sized> {
    inner: SortingResultKind<T>,
    config: &'a mut C
}

impl<'a, T: Ord + Copy + Sync + Send + Debug + Clone + PartialEq, C: ?Sized> SortingResult<'a, T, C> {

    pub fn get_inner_sorting_result_kind(&self) -> &SortingResultKind<T> {
        &self.inner
    }

    pub fn get_inner_config(&'a self) -> &'a C {
        self.config
    }

}

impl<'a, T: Ord + Copy + Sync + Send + Debug + Clone + PartialEq, C: ?Sized> From<SortingResult<'a, T, C>> for SortingResultKind<T> {
    fn from(value: SortingResult<'a, T, C>) -> Self {
        value.inner
    }
}

trait Sorting<'a, 'b, T: Ord + Copy + Sync + Send + Debug + Clone + PartialEq, C: ?Sized> {

    fn new(data: &'a [T], config: &'b mut C) -> Self where Self: Sized;

    fn sorting_get_sorted_result(self) -> SortingResult<'b, T, C>;

}


pub struct BogoSorting<'a, 'b, O: Optimizer, T: Ord + Copy + Sync + Send + Debug + Clone + PartialEq> {
    population: Vec<Vec<T>>,
    data: &'a [T],
    config: &'b mut BogoConfig<O, T>
}

impl<'a, 'b, O: Optimizer, T: Ord + Copy + Sync + Send + Debug + Clone + PartialEq> Sorting<'a, 'b, T, BogoConfig<O, T>> for BogoSorting<'a, 'b, O, T> {

    fn new(data: &'a [T], config: &'b mut BogoConfig<O, T>) -> Self {
        Self {
            population: Vec::with_capacity(0), // We assume that data is empty most of the time
            data,
            config,
        }
    }

    fn sorting_get_sorted_result(mut self) -> SortingResult<'b, T, BogoConfig<O, T>> {
        if self.data.is_empty() {
            return SortingResult {
                inner: SortingResultKind::SortingWasOkKindResult(vec![]),
                config: self.config,
            }
        }

        self.population = vec![self.data.to_vec(); self.config.population_size];

        let mut last_best_score = 0.0;
        let mut last_best_i = 0;
        let mut c = 0;

        self.config.optimizer.start(self.data.len());
        while last_best_score != (self.data.len() as f64) && c != self.config.give_up_count {
            let done_fraction = last_best_score / self.data.len() as f64;
            let mutation_chance = self.config.optimizer.get_mutation_chance(done_fraction);
            mutate(&mut self.population, mutation_chance);
            (last_best_score, last_best_i) = conform_to_best(&mut self.population, last_best_score, last_best_i, self.config.scoring_function);

            if self.config.verbose {
                c += 1;
                if c % 1000 == 0 || c < 10 {
                    println!("[{}] {:0.1}/{} {:0.2}% {:0.6}", c, last_best_score, self.data.len(), done_fraction * 100.0, mutation_chance);
                }
            }
        };

        if c == self.config.give_up_count {
            return SortingResult {
                inner: SortingResultKind::SortingWasIncompleteKindResult(self.population.pop().unwrap()),
                config: self.config,
            };
        }
        SortingResult {
            inner: SortingResultKind::SortingWasOkKindResult(self.population.pop().unwrap()),
            config: self.config,
        }
    }

}


#[allow(unused_imports)]
mod test {
    use super::*;

    #[test]
    fn test() {
        let n = 1000;
        let mut vec = Vec::with_capacity(n);
        for i in 0..n {
            vec.push(i);
        }
        // Shuffle
        for _ in 0..100 {
            mutate_one(&mut vec, 1.0);
        }
        // Config
        let simple = SimpleOptimizer::new(0.22, 1.3);
        let optimizer = ComplexOptimizer::new(simple, 0.00001, 2.5);
        let mut config = BogoConfig {
            give_up_count: 400000,
            verbose: true,
            population_size: 80,
            optimizer,
            scoring_function: score_one
        };
        // Sort
        let sorting = BogoSorting::new(&vec, &mut config);
        let result = sorting.sorting_get_sorted_result();
        let result_kind : SortingResultKind<_> = SortingResultKind::from(result);
        let ok_result = result_kind.unwrap_sorting_result_kind_with_incomplete();
        // Evaluate test
        let mut test_sorted = vec.clone();
        test_sorted.sort();
        assert_eq!(ok_result, Some(test_sorted));
    }

}