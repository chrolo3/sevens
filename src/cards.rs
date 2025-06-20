use crate::spades::{
    SevensSpades, SpadeFirstStrategy, SpadeLastRandom, SpadesLastHighest, SpadesRandom,
};
use crate::vanilla::{HighestFirst, LowestFirst, Sevens, VanillaRandom};
use colored::*;
use rand::prelude::ThreadRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suit {
    name: String,
    symbol: String,
    color: SuitColor,
    pub(crate) rank: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SuitColor {
    Red,
    Gray,
    Custom(u8, u8, u8),
}

impl Suit {
    pub fn new(name: &str, symbol: &str, color: SuitColor, rank: usize) -> Self {
        Suit {
            name: name.to_string(),
            symbol: symbol.to_string(),
            color,
            rank,
        }
    }

    pub fn to_string(&self) -> String {
        match &self.color {
            SuitColor::Red => self.symbol.red().to_string(),
            SuitColor::Gray => self.symbol.truecolor(192, 192, 192).to_string(),
            SuitColor::Custom(r, g, b) => self.symbol.truecolor(*r, *g, *b).to_string(),
        }
    }

    pub fn cmp(&self, other: &Suit) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// Helper function to generate base-26 (alphabetical) suit names
fn generate_alphabetical_label(mut number: usize) -> String {
    let mut label = String::new();
    number += 1; // Adjust 0-based indexing
    while number > 0 {
        let remainder = (number - 1) % 26;
        label.push((b'a' + remainder as u8) as char); // Convert to 'a'...'z'
        number = (number - 1) / 26;
    }
    label.chars().rev().collect()
}

pub fn alpha_suits(suit_count: usize) -> Vec<Suit> {
    let mut suits = Vec::with_capacity(suit_count);
    let mut rng = rand::rng();

    for i in 0..suit_count {
        // Generate name and symbol using the helper function
        let letter = generate_alphabetical_label(i);
        let name = format!("Suit {}", letter);
        let symbol = letter;

        // Generate bright color for dark themes
        let color = SuitColor::Custom(
            rng.random_range(128..=255), // Red
            rng.random_range(128..=255), // Green
            rng.random_range(128..=255), // Blue
        );

        let rank = i;
        let suit = Suit::new(&name, &symbol, color, rank);
        suits.push(suit);
    }

    suits
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Card {
    pub(crate) suit: Suit,
    pub(crate) value: usize,
}

impl Card {
    fn new(suit: Suit, value: usize) -> Card {
        Card { suit, value }
    }

    fn to_string(&self) -> String {
        format!("{}{}", self.suit.to_string(), self.value)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Board {
    pub(crate) suits: Vec<Suit>,
    pub(crate) ranges: Vec<(usize, usize)>,
    pub(crate) n7: usize,
}

pub(crate) trait GameRules {
    fn can_play(&self, board: &Board, card: &Card) -> bool;
    fn find_options(&self, board: &Board) -> usize;
}

#[derive(Debug, Clone)]
pub(crate) struct GameBoard<R: GameRules> {
    pub(crate) board: Board,
    rules: R,
}

impl<R: GameRules> GameBoard<R> {
    fn new(suits: Vec<Suit>, n7: usize, rules: R) -> Self {
        GameBoard {
            board: Board {
                ranges: vec![(0, 0); suits.len()],
                suits,
                n7,
            },
            rules,
        }
    }

    pub(crate) fn can_play(&self, card: &Card) -> bool {
        self.rules.can_play(&self.board, card)
    }

    fn play_card(&mut self, card: &Card) {
        let range = self.board.ranges[card.suit.rank];
        let n7 = self.board.n7;

        match range {
            (bottom, top) if bottom > 0 && top > 0 => {
                if card.value == n7 {
                    self.board.ranges[card.suit.rank] = (n7, n7);
                } else if card.value == bottom - 1 {
                    self.board.ranges[card.suit.rank] = (bottom - 1, top);
                } else if card.value == top + 1 {
                    self.board.ranges[card.suit.rank] = (bottom, top + 1);
                }
            }
            // Handle the case when there's no range yet
            (0, 0) => {
                if card.value == n7 {
                    self.board.ranges[card.suit.rank] = (n7, n7);
                }
            }
            // Handle potential partial ranges (should not happen in your game logic)
            (bottom, 0) if bottom > 0 => {
                if card.value == n7 {
                    self.board.ranges[card.suit.rank] = (n7, n7);
                } else if card.value == bottom - 1 {
                    self.board.ranges[card.suit.rank] = (bottom - 1, bottom);
                } else if card.value == bottom + 1 {
                    self.board.ranges[card.suit.rank] = (bottom, bottom + 1);
                }
            }
            (0, top) if top > 0 => {
                if card.value == n7 {
                    self.board.ranges[card.suit.rank] = (n7, n7);
                } else if card.value == top - 1 {
                    self.board.ranges[card.suit.rank] = (top - 1, top);
                } else if card.value == top + 1 {
                    self.board.ranges[card.suit.rank] = (top, top + 1);
                }
            }
            _ => {} // Handle any other cases (should not happen)
        }
    }
}

pub(crate) type Hand = Vec<Card>;

fn order_hand(hand: &mut Hand, n7: usize) {
    hand.sort_by(|a, b| {
        a.suit
            .cmp(&b.suit)
            .then(
                (a.value as i32 - n7 as i32)
                    .abs()
                    .cmp(&(b.value as i32 - n7 as i32).abs()),
            )
            .then(a.value.cmp(&b.value))
    });
}

pub(crate) trait Strategy: Debug {
    type Rules: GameRules;
    fn select_card(
        &self,
        hand: &Hand,
        board: &GameBoard<Self::Rules>,
        rng: &mut ThreadRng,
    ) -> Option<usize>;
}

#[derive(Debug, Clone)]
pub struct GameResult {
    winner: Option<usize>,
    board: Board,
    avg_options: f64,
    min_options: usize,
    max_options: usize,
}

fn play_game<R: GameRules + 'static>(
    strategies: &[&dyn Strategy<Rules = R>],
    rules: R,
    player_count: usize,
    suit_count: usize,
    n_size: usize,
) -> GameResult {
    let print_output = false;
    let suits = alpha_suits(suit_count);
    let n7 = n_size / 2;

    // Create cards
    let mut deck: Vec<Card> = Vec::with_capacity(suit_count * n_size);
    for suit in &suits {
        for value in 1..=n_size {
            deck.push(Card::new(suit.clone(), value));
        }
    }

    let mut rng = rand::rng();
    deck.shuffle(&mut rng);

    let mut hands: Vec<Hand> = vec![Vec::new(); player_count];
    for (index, card) in deck.iter().enumerate() {
        hands[index % player_count].push(card.clone());
    }

    for hand in hands.iter_mut() {
        order_hand(hand, n7);
    }

    // Find starting player (one with 7 of Spades)
    let mut starting_player: usize = 0;
    for (index, hand) in hands.iter().enumerate() {
        if hand
            .iter()
            .any(|card| card.suit.name == "Spades" && card.value == n7)
        {
            starting_player = index;
            break;
        }
    }

    let mut rng = rand::rng();

    // Create the game board with the provided rules
    let mut game_board = GameBoard::new(suits, n7, rules);

    // Statistics tracking for player options
    let mut total_options = 0;
    let mut min_options = usize::MAX;
    let mut max_options = 0;
    let mut turns_count = 0;

    let max_loops = deck.len() * player_count;
    let mut current_loop = starting_player;
    while current_loop < max_loops {
        let current_player = current_loop % player_count;

        // Calculate options available for the current player
        let options_count = game_board.rules.find_options(&game_board.board);

        // Update statistics
        total_options += options_count;
        min_options = min_options.min(options_count);
        max_options = max_options.max(options_count);
        turns_count += 1;

        if let Some(card_index) =
            strategies[current_player].select_card(&hands[current_player], &game_board, &mut rng)
        {
            let card = hands[current_player][card_index].clone();

            if game_board.can_play(&card) {
                game_board.play_card(&card);
                hands[current_player].remove(card_index);

                if print_output {
                    println!("{} plays {}", current_player, card.to_string());
                }

                if hands[current_player].is_empty() {
                    let avg_options = if turns_count > 0 {
                        total_options as f64 / turns_count as f64
                    } else {
                        0.0
                    };
                    let min_options = if min_options == usize::MAX {
                        0
                    } else {
                        min_options
                    };

                    return GameResult {
                        winner: Some(current_player),
                        board: game_board.board,
                        avg_options,
                        min_options,
                        max_options,
                    };
                }
            } else {
                if print_output {
                    println!("{} cannot play card", current_player);
                }
            }
        } else {
            if print_output {
                println!("{} cannot play card", current_player);
            }
        }
        current_loop += 1;
    }

    let avg_options = if turns_count > 0 {
        total_options as f64 / turns_count as f64
    } else {
        0.0
    };
    let min_options = if min_options == usize::MAX {
        0
    } else {
        min_options
    };

    GameResult {
        winner: None,
        board: game_board.board,
        avg_options,
        min_options,
        max_options,
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct StrategyPerformanceData {
    suit_count: usize,
    n_size: usize,
    deck_size: usize,
    strategy_win_rates: HashMap<String, f64>,
    win_rate_variance: f64,
    max_win_rate_diff: f64,
    avg_options_per_turn: f64,
    strategy_advantage_factor: f64,
    total_avg_options: f64,
    min_options: usize,
    max_options: usize,
    total_min_options: f64,
    total_max_options: f64,
}

struct SimulationStatistics {
    games_completed: usize,
    player_wins: Vec<usize>,
    strategy_names: Vec<String>,
    total_avg_options: f64,
    total_min_options: usize,
    total_max_options: usize,
    min_options_ever: usize,
    max_options_ever: usize,
    size_of_deck: usize,
    suit_count: usize,
    n_size: usize,
}

impl SimulationStatistics {
    fn new(
        player_count: usize,
        size_of_deck: usize,
        suit_count: usize,
        n_size: usize,
        strategy_names: Vec<String>,
    ) -> SimulationStatistics {
        SimulationStatistics {
            games_completed: 0,
            player_wins: vec![0; player_count],
            strategy_names,
            total_avg_options: 0.0,
            total_min_options: 0,
            total_max_options: 0,
            min_options_ever: usize::MAX,
            max_options_ever: 0,
            size_of_deck,
            suit_count,
            n_size,
        }
    }

    fn record_result(&mut self, result: &GameResult) {
        self.games_completed += 1;

        // Record winner if there is one
        if let Some(winner) = result.winner {
            self.player_wins[winner] += 1;
        }

        // Record options statistics
        self.total_avg_options += result.avg_options;
        self.total_min_options += result.min_options;
        self.total_max_options += result.max_options;
        self.min_options_ever = self.min_options_ever.min(result.min_options);
        self.max_options_ever = self.max_options_ever.max(result.max_options);
    }

    fn calculate_win_rate_variance(&self) -> f64 {
        let win_percentages: Vec<f64> = self
            .player_wins
            .iter()
            .map(|&wins| wins as f64 / self.games_completed as f64)
            .collect();

        let mean = win_percentages.iter().sum::<f64>() / win_percentages.len() as f64;

        let variance = win_percentages
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>()
            / win_percentages.len() as f64;

        variance
    }

    // Calculate the difference between the best and worst performing strategies
    fn calculate_max_win_rate_diff(&self) -> f64 {
        let win_percentages: Vec<f64> = self
            .player_wins
            .iter()
            .map(|&wins| wins as f64 / self.games_completed as f64)
            .collect();

        let max = win_percentages.iter().fold(f64::MIN, |a, &b| a.max(b));
        let min = win_percentages.iter().fold(f64::MAX, |a, &b| a.min(b));

        max - min
    }

    // Calculate a strategy advantage factor - a measure of how much strategy choice impacts outcomes
    fn calculate_strategy_advantage_factor(&self) -> f64 {
        // Normalized variance - scales with the number of options per turn
        // This helps quantify if strategy matters more when there are more or fewer options
        let variance = self.calculate_win_rate_variance();
        let avg_options = self.total_avg_options / self.games_completed as f64;

        // Scale factor that decreases as the number of suits increases
        let scale = 1.0 / (1.0 + (self.suit_count as f64).ln());

        variance * avg_options * scale
    }

    fn get_performance_data(&self) -> StrategyPerformanceData {
        let strategy_win_rates = self
            .strategy_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let win_rate = if self.games_completed > 0 {
                    self.player_wins[i] as f64 / self.games_completed as f64
                } else {
                    0.0
                };
                let player_label = format!("{}{}", name, i);
                (player_label, win_rate)
            })
            .collect();

        StrategyPerformanceData {
            suit_count: self.suit_count,
            n_size: self.n_size,
            deck_size: self.size_of_deck,
            strategy_win_rates,
            win_rate_variance: self.calculate_win_rate_variance(),
            max_win_rate_diff: self.calculate_max_win_rate_diff(),
            avg_options_per_turn: if self.games_completed > 0 {
                self.total_avg_options / self.games_completed as f64
            } else {
                0.0
            },
            strategy_advantage_factor: self.calculate_strategy_advantage_factor(),
            total_avg_options: self.total_avg_options / 10000.0,
            min_options: self.min_options_ever,
            max_options: self.max_options_ever,
            total_min_options: self.total_min_options as f64 / 10000.0,
            total_max_options: self.total_max_options as f64 / 10000.0,
        }
    }
}

pub(crate) fn run_simulations_sevens(player_count: usize, simulation_count: usize) {
    // Define all available strategies
    let strategy_index_combinations = vec![
        vec![0, 0, 0, 0],
        vec![1, 1, 1, 1],
        vec![2, 2, 2, 2],
        vec![0, 0, 0, 1],
        vec![0, 0, 1, 1],
        vec![0, 1, 1, 1],
        vec![0, 0, 0, 2],
        vec![0, 0, 2, 2],
        vec![0, 2, 2, 2],
        vec![1, 1, 1, 2],
        vec![1, 1, 2, 2],
        vec![1, 2, 2, 2],
        vec![0, 1, 1, 2],
        vec![0, 1, 2, 2],
    ];

    println!(
        "Testing {} different strategy combinations",
        strategy_index_combinations.len()
    );

    let suit_counts: Vec<usize> = vec![
        4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 48, 48, 48, 48, 48, 64, 64, 64, 64, 64,
    ];
    let suit_n_size: Vec<usize> = vec![
        13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111,
    ];
    let all_performance_data = Mutex::new(Vec::new());

    suit_counts
        .iter()
        .zip(suit_n_size.iter())
        .for_each(|(suit_count, n_size)| {
            let combo_results: Vec<StrategyPerformanceData> = strategy_index_combinations
                .par_iter()
                .enumerate() // This should work with par_iter
                .map(|(_, strategy_index_combination)| {
                    let strategies: Vec<&dyn Strategy<Rules = Sevens>> =
                        vec![&VanillaRandom, &LowestFirst, &HighestFirst];

                    let strategy_assignment: Vec<&dyn Strategy<Rules = Sevens>> =
                        strategy_index_combination
                            .iter()
                            .map(|&index| strategies[index])
                            .collect();

                    let combo_strategy_names: Vec<String> = strategy_assignment
                        .iter()
                        .map(|s| format!("{:?}", s))
                        .collect();

                    let size_of_deck = suit_count * n_size;

                    let mut stats = SimulationStatistics::new(
                        player_count,
                        size_of_deck,
                        *suit_count,
                        *n_size,
                        combo_strategy_names,
                    );

                    for _ in 0..simulation_count {
                        let result = play_game(
                            &strategy_assignment,
                            Sevens,
                            player_count,
                            *suit_count,
                            *n_size,
                        );

                        stats.record_result(&result);

                    }

                    stats.get_performance_data()
                })
                .collect();

            all_performance_data.lock().unwrap().extend(combo_results);
        });

    save_strategy_performance_data(&all_performance_data, "sevens");

    let final_results = all_performance_data.into_inner().unwrap();

    analyze_game_statistics(&final_results, "sevens");
}

pub(crate) fn run_simulations_spades(player_count: usize, simulation_count: usize) {
    // Define all available strategies
    let strategy_index_combinations = vec![
        // All players using the same strategy (baselines)
        vec![0, 0, 0, 0], // All random (strategy 0)
        vec![1, 1, 1, 1], // All strategy 1
        vec![2, 2, 2, 2], // All strategy 2
        vec![3, 3, 3, 3], // All strategy 3
        // Strategy 0 vs Strategy 1 combinations
        vec![0, 0, 0, 1], // 3 strategy 0, 1 strategy 1
        vec![0, 0, 1, 1], // 2 strategy 0, 2 strategy 1
        vec![0, 1, 1, 1], // 1 strategy 0, 3 strategy 1
        // Strategy 0 vs Strategy 2 combinations
        vec![0, 0, 0, 2], // 3 strategy 0, 1 strategy 2
        vec![0, 0, 2, 2], // 2 strategy 0, 2 strategy 2
        vec![0, 2, 2, 2], // 1 strategy 0, 3 strategy 2
        // Strategy 0 vs Strategy 3 combinations
        vec![0, 0, 0, 3], // 3 strategy 0, 1 strategy 3
        vec![0, 0, 3, 3], // 2 strategy 0, 2 strategy 3
        vec![0, 3, 3, 3], // 1 strategy 0, 3 strategy 3
        // Strategy 1 vs Strategy 2 combinations
        vec![1, 1, 1, 2], // 3 strategy 1, 1 strategy 2
        vec![1, 1, 2, 2], // 2 strategy 1, 2 strategy 2
        vec![1, 2, 2, 2], // 1 strategy 1, 3 strategy 2
        // Strategy 1 vs Strategy 3 combinations
        vec![1, 1, 1, 3], // 3 strategy 1, 1 strategy 3
        vec![1, 1, 3, 3], // 2 strategy 1, 2 strategy 3
        vec![1, 3, 3, 3], // 1 strategy 1, 3 strategy 3
        // Strategy 2 vs Strategy 3 combinations
        vec![2, 2, 2, 3], // 3 strategy 2, 1 strategy 3
        vec![2, 2, 3, 3], // 2 strategy 2, 2 strategy 3
        vec![2, 3, 3, 3], // 1 strategy 2, 3 strategy 3
        // Mixed strategy combinations (three different strategies)
        vec![0, 1, 1, 2], // 1 strategy 0, 2 strategy 1, 1 strategy 2
        vec![0, 1, 2, 2], // 1 strategy 0, 1 strategy 1, 2 strategy 2
        vec![0, 0, 1, 2], // 2 strategy 0, 1 strategy 1, 1 strategy 2
        vec![0, 1, 1, 3], // 1 strategy 0, 2 strategy 1, 1 strategy 3
        vec![0, 1, 3, 3], // 1 strategy 0, 1 strategy 1, 2 strategy 3
        vec![0, 0, 1, 3], // 2 strategy 0, 1 strategy 1, 1 strategy 3
        vec![0, 2, 2, 3], // 1 strategy 0, 2 strategy 2, 1 strategy 3
        vec![0, 2, 3, 3], // 1 strategy 0, 1 strategy 2, 2 strategy 3
        vec![0, 0, 2, 3], // 2 strategy 0, 1 strategy 2, 1 strategy 3
        vec![1, 2, 2, 3], // 1 strategy 1, 2 strategy 2, 1 strategy 3
        vec![1, 2, 3, 3], // 1 strategy 1, 1 strategy 2, 2 strategy 3
        vec![1, 1, 2, 3], // 2 strategy 1, 1 strategy 2, 1 strategy 3
        // Four different strategies in one game (positional testing)
        vec![0, 1, 2, 3], // All four strategies present
    ];

    println!(
        "Testing {} different strategy combinations",
        strategy_index_combinations.len()
    );

    let suit_counts: Vec<usize> = vec![
        4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 48, 48, 48, 48, 48, 64, 64, 64, 64, 64,
    ];
    let suit_n_size: Vec<usize> = vec![
        13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111, 13, 27, 55, 83, 111,
    ];
    let all_performance_data = Mutex::new(Vec::new());

    suit_counts
        .iter()
        .zip(suit_n_size.iter())
        .for_each(|(suit_count, n_size)| {
            let combo_results: Vec<StrategyPerformanceData> = strategy_index_combinations
                .par_iter()
                .enumerate() // This should work with par_iter
                .map(|(_, strategy_index_combination)| {

                    let strategies: Vec<&dyn Strategy<Rules = SevensSpades>> = vec![
                        &SpadesRandom,
                        &SpadeFirstStrategy,
                        &SpadeLastRandom,
                        &SpadesLastHighest,
                    ];

                    let strategy_assignment: Vec<&dyn Strategy<Rules = SevensSpades>> =
                        strategy_index_combination
                            .iter()
                            .map(|&index| strategies[index])
                            .collect();

                    let combo_strategy_names: Vec<String> = strategy_assignment
                        .iter()
                        .map(|s| format!("{:?}", s))
                        .collect();

                    let size_of_deck = suit_count * n_size;

                    let mut stats = SimulationStatistics::new(
                        player_count,
                        size_of_deck,
                        *suit_count,
                        *n_size,
                        combo_strategy_names,
                    );

                    for _ in 0..simulation_count {
                        let result = play_game(
                            &strategy_assignment,
                            SevensSpades,
                            player_count,
                            *suit_count,
                            *n_size,
                        );

                        stats.record_result(&result);
                    }

                    stats.get_performance_data()
                })
                .collect();

            all_performance_data.lock().unwrap().extend(combo_results);
        });

    save_strategy_performance_data(&all_performance_data, "spades");

    let final_results = all_performance_data.into_inner().unwrap();

    analyze_game_statistics(&final_results, "spades");
}

fn save_strategy_performance_data(data: &Mutex<Vec<StrategyPerformanceData>>, name: &str) {
    let json = serde_json::to_string_pretty(data).unwrap();
    let file_name = format!("{}_strategy_performance_data.json", name);
    std::fs::write(file_name, json)
        .expect("Failed to write strategy performance data to file");
    println!("Strategy performance data saved to 'strategy_performance_data.json'");
}

fn analyze_game_statistics(data: &[StrategyPerformanceData], name: &str) {
    println!("\n=== GAME STATISTICS ANALYSIS ({}) ===", name);

    // Group data by suit count
    let mut by_suit_count: HashMap<usize, Vec<&StrategyPerformanceData>> = HashMap::new();
    for item in data {
        by_suit_count.entry(item.suit_count).or_default().push(item);
    }

    // Calculate statistics for each suit count
    let mut game_stats: Vec<(usize, usize, f64, f64, f64, f64, f64, f64, f64)> = by_suit_count
        .iter()
        .map(|(&suit_count, items)| {
            let configurations = items.len();
            let total_games = configurations * 10000; // 10,000 games per configuration
            let avg_deck_size = items.iter().map(|i| i.deck_size).sum::<usize>() as f64 / items.len() as f64;
            let avg_options_per_turn = items.iter().map(|i| i.avg_options_per_turn).sum::<f64>() / items.len() as f64;
            let avg_min_options = items.iter().map(|i| i.min_options).sum::<usize>() as f64 / items.len() as f64;
            let avg_max_options = items.iter().map(|i| i.max_options).sum::<usize>() as f64 / items.len() as f64;
            let avg_strategy_advantage = items.iter().map(|i| i.strategy_advantage_factor).sum::<f64>() / items.len() as f64;
            let avg_win_rate_variance = items.iter().map(|i| i.win_rate_variance).sum::<f64>() / items.len() as f64;
            let avg_max_win_diff = items.iter().map(|i| i.max_win_rate_diff).sum::<f64>() / items.len() as f64;

            (
                suit_count,
                total_games,
                avg_deck_size,
                avg_options_per_turn,
                avg_min_options,
                avg_max_options,
                avg_strategy_advantage,
                avg_win_rate_variance,
                avg_max_win_diff,
            )
        })
        .collect();

    // Sort by suit count
    game_stats.sort_by_key(|&(count, _, _, _, _, _, _, _, _)| count);

    // Print comprehensive table
    println!("\n📊 DETAILED STATISTICS BY SUIT COUNT");
    println!("| Suits | Games | Configs | Deck | Avg Opts | Min | Max | Strat Adv | Variance | Max Diff |");
    println!("|-------|-------|---------|------|----------|-----|-----|-----------|----------|----------|");

    for (suit_count, total_games, deck_size, avg_options, min_opts, max_opts, strat_adv, variance, max_diff) in &game_stats {
        let configurations = total_games / 10000;
        println!(
            "| {:5} | {:5}k | {:7} | {:4.0} | {:8.2} | {:3.0} | {:3.0} | {:9.3} | {:8.3} | {:8.2}% |",
            suit_count,
            total_games / 1000,
            configurations,
            deck_size,
            avg_options,
            min_opts,
            max_opts,
            strat_adv,
            variance,
            max_diff * 100.0
        );
    }

    // Calculate overall statistics
    let total_games: usize = game_stats.iter().map(|(_, games, _, _, _, _, _, _, _)| games).sum();
    let total_configurations: usize = game_stats.iter().map(|(_, games, _, _, _, _, _, _, _)| games / 10000).sum();
    let avg_options_overall: f64 = game_stats
        .iter()
        .map(|(_, games, _, avg_opts, _, _, _, _, _)| avg_opts * (*games as f64))
        .sum::<f64>() / total_games as f64;

    println!("\n🎯 OVERALL PERFORMANCE METRICS");
    println!("Total strategy configurations tested: {:}", total_configurations);
    println!("Total games simulated: {:} ({:} games per config)", total_games, 10000);
    println!("Average options per turn (weighted): {:.2}", avg_options_overall);

    // Game complexity analysis
    if let (Some(min_complexity), Some(max_complexity)) = (
        game_stats.iter().min_by(|a, b| a.3.partial_cmp(&b.3).unwrap()),
        game_stats.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()),
    ) {
        println!("\n🧠 GAME COMPLEXITY ANALYSIS");
        println!("Simplest configuration: {} suits ({:.2} avg options/turn)", min_complexity.0, min_complexity.3);
        println!("Most complex configuration: {} suits ({:.2} avg options/turn)", max_complexity.0, max_complexity.3);

        let complexity_ratio = max_complexity.3 / min_complexity.3;
        println!("Complexity multiplier: {:.2}x increase", complexity_ratio);

        let option_range_simple = max_complexity.5 - min_complexity.4;
        let option_range_complex = max_complexity.5 - max_complexity.4;
        println!("Decision range: {:.0} options (simple) vs {:.0} options (complex)",
                 option_range_simple, option_range_complex);
    }

    // Strategy effectiveness analysis
    println!("\n⚔️ STRATEGY EFFECTIVENESS ANALYSIS");
    let avg_strategy_advantage: f64 = game_stats.iter()
                                                .map(|(_, games, _, _, _, _, adv, _, _)| adv * (*games as f64))
                                                .sum::<f64>() / total_games as f64;

    let avg_variance: f64 = game_stats.iter()
                                      .map(|(_, games, _, _, _, _, _, var, _)| var * (*games as f64))
                                      .sum::<f64>() / total_games as f64;

    let avg_max_diff: f64 = game_stats.iter()
                                      .map(|(_, games, _, _, _, _, _, _, diff)| diff * (*games as f64))
                                      .sum::<f64>() / total_games as f64;

    println!("Average strategy advantage factor: {:.3}", avg_strategy_advantage);
    println!("Average win rate variance: {:.3}", avg_variance);
    println!("Average max win rate difference: {:.2}%", avg_max_diff * 100.0);

    if avg_strategy_advantage > 1.5 {
        println!("🟢 High strategy differentiation - skill matters significantly");
    } else if avg_strategy_advantage > 1.2 {
        println!("🟡 Moderate strategy differentiation - some skill advantage");
    } else {
        println!("🔴 Low strategy differentiation - mostly luck-based");
    }

    // Performance trends
    println!("\n📈 TREND ANALYSIS");
    let complexity_trend = game_stats.windows(2)
                                     .map(|w| if w[1].3 > w[0].3 { 1 } else if w[1].3 < w[0].3 { -1 } else { 0 })
                                     .sum::<i32>();

    let strategy_trend = game_stats.windows(2)
                                   .map(|w| if w[1].6 > w[0].6 { 1 } else if w[1].6 < w[0].6 { -1 } else { 0 })
                                   .sum::<i32>();

    match complexity_trend {
        x if x > 0 => println!("🔺 Complexity increases with more suits (+{})", x),
        x if x < 0 => println!("🔻 Complexity decreases with more suits ({})", x),
        _ => println!("➡️ Complexity remains stable across suit counts"),
    }

    match strategy_trend {
        x if x > 0 => println!("🔺 Strategy importance increases with more suits (+{})", x),
        x if x < 0 => println!("🔻 Strategy importance decreases with more suits ({})", x),
        _ => println!("➡️ Strategy importance remains stable across suit counts"),
    }

    // Computational cost analysis
    println!("\n💻 COMPUTATIONAL COST ANALYSIS");
    let mut total_estimated_decisions = 0.0;
    for (suit_count, _, deck_size, avg_options, _, _, _, _, _) in &game_stats {
        let estimated_turns_per_game = deck_size * 0.75; // Rough estimate
        let decisions_per_game = estimated_turns_per_game * avg_options;
        let total_decisions_10k = decisions_per_game * 10000.0;
        total_estimated_decisions += total_decisions_10k;

        let estimated_seconds = total_decisions_10k / 1000.0; // Assuming 1ms per decision
        let estimated_minutes = estimated_seconds / 60.0;

        println!("{} suits: {:8.0} decisions, {:6.1}s ({:4.1} min) per 10k games",
                 suit_count, total_decisions_10k, estimated_seconds, estimated_minutes);
    }

    println!("\nTotal estimated decisions across all simulations: {:.0}", total_estimated_decisions);
    println!("Estimated total computation time: {:.1} hours", total_estimated_decisions / 1000.0 / 3600.0);

    // Performance recommendations
    println!("\n💡 PERFORMANCE INSIGHTS");

    if let Some(sweet_spot) = game_stats.iter()
                                        .max_by(|a, b| (a.6 / a.3).partial_cmp(&(b.6 / b.3)).unwrap()) {
        println!("🎯 Optimal complexity/strategy ratio: {} suits", sweet_spot.0);
    }

    if avg_max_diff > 0.15 {
        println!("⚠️ High strategy impact detected - consider strategy balancing");
    }

    if complexity_trend > 2 {
        println!("📊 Strong complexity scaling - good for difficulty progression");
    }

    println!("{}", "\n".to_owned() + "=".repeat(70).as_str());
}
