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
    let mut rng = rand::thread_rng();

    for i in 0..suit_count {
        // Generate name and symbol using the helper function
        let letter = generate_alphabetical_label(i);
        let name = format!("Suit {}", letter);
        let symbol = letter;

        // Generate bright color for dark themes
        let color = SuitColor::Custom(
            rng.gen_range(128..=255), // Red
            rng.gen_range(128..=255), // Green
            rng.gen_range(128..=255), // Blue
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

    pub fn to_string(&self) -> String {
        let mut board_string = String::new();
        for suit in self.board.suits.iter() {
            board_string.push_str(&format!(
                "{} : {}-{}\n",
                suit.to_string(),
                self.board.ranges[suit.rank].0,
                self.board.ranges[suit.rank].1
            ));
        }
        board_string
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

// Add this function to format a hand as a string
fn hand_to_string(hand: &[Card]) -> String {
    hand.iter()
        .map(|card| card.to_string())
        .collect::<Vec<String>>()
        .join(" ")
}

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

    let mut rng = rand::thread_rng();
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

    let mut rng = rand::thread_rng();

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
    total_cards_left: usize,
    player_wins: Vec<usize>,
    strategy_names: Vec<String>, // Store strategy names for reference
    total_avg_options: f64,
    total_min_options: usize,
    total_max_options: usize,
    min_options_ever: usize,
    max_options_ever: usize,
    size_of_deck: usize,
    n7: usize,
    player_count: usize,
    player_hand_size: usize,
    suit_count: usize,             // New field to track the number of suits
    n_size: usize,                 // New field to track n_size
    strategy_advantage_score: f64, // New field to track strategy advantage
}

impl SimulationStatistics {
    fn new(
        player_count: usize,
        size_of_deck: usize,
        suit_count: usize,
        n_size: usize,
        strategy_names: Vec<String>,
    ) -> SimulationStatistics {
        let player_hand_size = size_of_deck / player_count;
        let n7 = n_size / 2;
        SimulationStatistics {
            games_completed: 0,
            total_cards_left: 0,
            player_wins: vec![0; player_count],
            strategy_names,
            total_avg_options: 0.0,
            total_min_options: 0,
            total_max_options: 0,
            min_options_ever: usize::MAX,
            max_options_ever: 0,
            size_of_deck,
            n7,
            player_count,
            player_hand_size,
            suit_count,
            n_size,
            strategy_advantage_score: 0.0,
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

    fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct Statistics {
            games_completed: usize,
            total_cards_left: usize,
            player_wins: Vec<usize>,
            size_of_deck: usize,
            n7: usize,
            player_count: usize,
            player_hand_size: usize,
            total_avg_options: f64,
            total_min_options: f64,
            total_max_options: f64,
            min_options_ever: usize,
            max_options_ever: usize,
        }

        let stats = Statistics {
            games_completed: self.games_completed,
            total_cards_left: self.total_cards_left,
            player_wins: self.player_wins.clone(),
            size_of_deck: self.size_of_deck,
            n7: self.n7,
            player_count: self.player_count,
            player_hand_size: self.player_hand_size,
            total_avg_options: self.total_avg_options / self.games_completed as f64,
            total_min_options: self.total_min_options as f64 / self.games_completed as f64,
            total_max_options: self.total_max_options as f64 / self.games_completed as f64,
            min_options_ever: if self.min_options_ever == usize::MAX {
                0
            } else {
                self.min_options_ever
            },
            max_options_ever: self.max_options_ever,
        };

        serde_json::to_string_pretty(&stats).expect("Failed to serialize statistics")
    }
}

pub(crate) fn run_simulations(player_count: usize, simulation_count: usize) {
    // Define all available strategies

    // Generate unique combinations of strategies (with repetition allowed)
    // let strategy_index_combinations =
    //     generate_strategy_combinations(player_count, strategies.len());

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

    // let strategy_index_combinations = vec![
    //     // All players using the same strategy (baselines)
    //     vec![0, 0, 0, 0], // All random (strategy 0)
    //     vec![1, 1, 1, 1], // All strategy 1
    //     vec![2, 2, 2, 2], // All strategy 2
    //     vec![3, 3, 3, 3], // All strategy 3
    //     // Strategy 0 vs Strategy 1 combinations
    //     vec![0, 0, 0, 1], // 3 strategy 0, 1 strategy 1
    //     vec![0, 0, 1, 1], // 2 strategy 0, 2 strategy 1
    //     vec![0, 1, 1, 1], // 1 strategy 0, 3 strategy 1
    //     // Strategy 0 vs Strategy 2 combinations
    //     vec![0, 0, 0, 2], // 3 strategy 0, 1 strategy 2
    //     vec![0, 0, 2, 2], // 2 strategy 0, 2 strategy 2
    //     vec![0, 2, 2, 2], // 1 strategy 0, 3 strategy 2
    //     // Strategy 0 vs Strategy 3 combinations
    //     vec![0, 0, 0, 3], // 3 strategy 0, 1 strategy 3
    //     vec![0, 0, 3, 3], // 2 strategy 0, 2 strategy 3
    //     vec![0, 3, 3, 3], // 1 strategy 0, 3 strategy 3
    //     // Strategy 1 vs Strategy 2 combinations
    //     vec![1, 1, 1, 2], // 3 strategy 1, 1 strategy 2
    //     vec![1, 1, 2, 2], // 2 strategy 1, 2 strategy 2
    //     vec![1, 2, 2, 2], // 1 strategy 1, 3 strategy 2
    //     // Strategy 1 vs Strategy 3 combinations
    //     vec![1, 1, 1, 3], // 3 strategy 1, 1 strategy 3
    //     vec![1, 1, 3, 3], // 2 strategy 1, 2 strategy 3
    //     vec![1, 3, 3, 3], // 1 strategy 1, 3 strategy 3
    //     // Strategy 2 vs Strategy 3 combinations
    //     vec![2, 2, 2, 3], // 3 strategy 2, 1 strategy 3
    //     vec![2, 2, 3, 3], // 2 strategy 2, 2 strategy 3
    //     vec![2, 3, 3, 3], // 1 strategy 2, 3 strategy 3
    //     // Mixed strategy combinations (three different strategies)
    //     vec![0, 1, 1, 2], // 1 strategy 0, 2 strategy 1, 1 strategy 2
    //     vec![0, 1, 2, 2], // 1 strategy 0, 1 strategy 1, 2 strategy 2
    //     vec![0, 0, 1, 2], // 2 strategy 0, 1 strategy 1, 1 strategy 2
    //     vec![0, 1, 1, 3], // 1 strategy 0, 2 strategy 1, 1 strategy 3
    //     vec![0, 1, 3, 3], // 1 strategy 0, 1 strategy 1, 2 strategy 3
    //     vec![0, 0, 1, 3], // 2 strategy 0, 1 strategy 1, 1 strategy 3
    //     vec![0, 2, 2, 3], // 1 strategy 0, 2 strategy 2, 1 strategy 3
    //     vec![0, 2, 3, 3], // 1 strategy 0, 1 strategy 2, 2 strategy 3
    //     vec![0, 0, 2, 3], // 2 strategy 0, 1 strategy 2, 1 strategy 3
    //     vec![1, 2, 2, 3], // 1 strategy 1, 2 strategy 2, 1 strategy 3
    //     vec![1, 2, 3, 3], // 1 strategy 1, 1 strategy 2, 2 strategy 3
    //     vec![1, 1, 2, 3], // 2 strategy 1, 1 strategy 2, 1 strategy 3
    //     // Four different strategies in one game (positional testing)
    //     vec![0, 1, 2, 3], // All four strategies present
    // ];

    println!(
        "Testing {} different strategy combinations",
        strategy_index_combinations.len()
    );

    let player_count = 4;
    let suit_counts: Vec<usize> = vec![
        4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64,
    ];
    let suit_n_size: Vec<usize> = vec![
        13, 27, 55, 111, 13, 27, 55, 111, 13, 27, 55, 111, 13, 27, 55, 111, 13, 27, 55, 111,
    ];
    let all_performance_data = Mutex::new(Vec::new());

    suit_counts
        .iter()
        .zip(suit_n_size.iter())
        .for_each(|(suit_count, n_size)| {
            let combo_results: Vec<StrategyPerformanceData> = strategy_index_combinations
                .par_iter()
                .enumerate() // This should work with par_iter
                .map(|(combo_index, strategy_index_combination)| {
                    let strategies: Vec<&dyn Strategy<Rules = Sevens>> =
                        vec![&VanillaRandom, &LowestFirst, &HighestFirst];

                    // let strategies: Vec<&dyn Strategy<Rules = SevensSpades>> = vec![
                    //     &SpadesRandom,
                    //     &SpadeFirstStrategy,
                    //     &SpadeLastRandom,
                    //     &SpadesLastHighest,
                    // ];

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
                    let mut ties = 0;

                    for _ in 0..simulation_count {
                        let result = play_game(
                            &strategy_assignment,
                            Sevens,
                            player_count,
                            *suit_count,
                            *n_size,
                        );

                        stats.record_result(&result);

                        if result.winner.is_none() {
                            ties += 1;
                        }
                    }

                    stats.get_performance_data()
                })
                .collect();

            all_performance_data.lock().unwrap().extend(combo_results);
        });

    save_strategy_performance_data(&all_performance_data);

    let final_results = all_performance_data.into_inner().unwrap();

    analyze_game_statistics(&final_results);
}

fn save_strategy_performance_data(data: &Mutex<Vec<StrategyPerformanceData>>) {
    let json = serde_json::to_string_pretty(data).unwrap();
    std::fs::write("strategy_performance_data.json", json)
        .expect("Failed to write strategy performance data to file");
    println!("Strategy performance data saved to 'strategy_performance_data.json'");
}

fn analyze_game_statistics(data: &[StrategyPerformanceData]) {
    println!("\n=== GAME STATISTICS ANALYSIS ===");

    // Group data by suit count
    let mut by_suit_count: HashMap<usize, Vec<&StrategyPerformanceData>> = HashMap::new();
    for item in data {
        by_suit_count.entry(item.suit_count).or_default().push(item);
    }

    // Calculate statistics for each suit count
    let mut game_stats: Vec<(usize, usize, f64, f64, f64, f64)> = by_suit_count
        .iter()
        .map(|(&suit_count, items)| {
            let configurations = items.len();
            let total_games = configurations * 10000; // 10,000 games per configuration
            let avg_deck_size =
                items.iter().map(|i| i.deck_size).sum::<usize>() as f64 / items.len() as f64;
            let avg_options_per_turn =
                items.iter().map(|i| i.avg_options_per_turn).sum::<f64>() / items.len() as f64;
            let avg_min_options =
                items.iter().map(|i| i.min_options).sum::<usize>() as f64 / items.len() as f64;
            let avg_max_options =
                items.iter().map(|i| i.max_options).sum::<usize>() as f64 / items.len() as f64;

            (
                suit_count,
                total_games,
                avg_deck_size,
                avg_options_per_turn,
                avg_min_options,
                avg_max_options,
            )
        })
        .collect();

    // Sort by suit count
    game_stats.sort_by_key(|&(count, _, _, _, _, _)| count);

    // Print table header
    println!("| Suits | Total Games | Configurations | Avg Deck Size | Avg Options/Turn | Min Options | Max Options |");
    println!("|-------|-------------|----------------|---------------|------------------|-------------|-------------|");

    // Print each row
    // Print each row
    for (suit_count, total_games, deck_size, avg_options, min_opts, max_opts) in &game_stats {
        let configurations = total_games / 10000;
        println!(
            "| {:5} | {:11} | {:14} | {:13.1} | {:16.2} | {:11.2} | {:11.2} |",
            suit_count, total_games, configurations, deck_size, avg_options, min_opts, max_opts
        );
    }

    // Calculate overall statistics
    let total_games: usize = game_stats.iter().map(|(_, games, _, _, _, _)| games).sum();
    let total_configurations: usize = game_stats
        .iter()
        .map(|(_, games, _, _, _, _)| games / 10000)
        .sum();
    let avg_options_overall: f64 = game_stats
        .iter()
        .map(|(_, games, _, avg_opts, _, _)| avg_opts * (*games as f64))
        .sum::<f64>()
        / total_games as f64;

    println!("\n=== OVERALL STATISTICS ===");
    println!(
        "Total strategy configurations tested: {}",
        total_configurations
    );
    println!(
        "Total games played: {} ({} games per configuration)",
        total_games, 10000
    );
    println!(
        "Average options per turn (weighted): {:.2}",
        avg_options_overall
    );

    // Calculate total simulation time estimate
    let total_decisions = total_games as f64 * avg_options_overall;
    println!("Estimated total decisions made: {:.0}", total_decisions);

    // Find configuration with most/least complexity
    if let (Some(min_complexity), Some(max_complexity)) = (
        game_stats
            .iter()
            .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap()),
        game_stats
            .iter()
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()),
    ) {
        println!("\nGame Complexity Analysis:");
        println!(
            "Simplest configuration: {} suits ({:.2} avg options/turn)",
            min_complexity.0, min_complexity.3
        );
        println!(
            "Most complex configuration: {} suits ({:.2} avg options/turn)",
            max_complexity.0, max_complexity.3
        );

        let complexity_ratio = max_complexity.3 / min_complexity.3;
        println!("Complexity ratio: {:.2}x", complexity_ratio);

        // Compare decision load between simplest and most complex
        let simple_decisions_per_10k = 10000.0 * min_complexity.3 * min_complexity.2 * 0.75;
        let complex_decisions_per_10k = 10000.0 * max_complexity.3 * max_complexity.2 * 0.75;
        println!(
            "Decision load difference: {:.0} vs {:.0} decisions per 10k games",
            simple_decisions_per_10k, complex_decisions_per_10k
        );
    }

    // Analyze trends
    println!("\n=== TREND ANALYSIS ===");
    let options_trend = game_stats
        .windows(2)
        .map(|w| {
            if w[1].3 > w[0].3 {
                1
            } else if w[1].3 < w[0].3 {
                -1
            } else {
                0
            }
        })
        .sum::<i32>();

    match options_trend {
        x if x > 0 => println!("✓ Game complexity generally increases with more suits"),
        x if x < 0 => println!("✓ Game complexity generally decreases with more suits"),
        _ => println!("~ Game complexity shows mixed trends across suit counts"),
    }

    // Estimate timing per 10,000 games (assuming 1ms per option consideration)
    println!("\n=== ESTIMATED TIMING PER 10,000 GAMES ===");
    for (suit_count, _, deck_size, avg_options, _, _) in &game_stats {
        let estimated_turns_per_game = deck_size * 0.75; // Rough estimate of turns per game
        let decisions_per_game = estimated_turns_per_game * avg_options;
        let total_decisions_10k = decisions_per_game * 10000.0;
        let estimated_ms = total_decisions_10k; // 1ms per decision
        let estimated_seconds = estimated_ms / 1000.0;
        let estimated_minutes = estimated_seconds / 60.0;

        println!(
            "{} suits: {:.0} decisions, {:.1}s ({:.1} min) per 10k games",
            suit_count, total_decisions_10k, estimated_seconds, estimated_minutes
        );
    }

    // Save statistics summary
    let stats_summary = serde_json::json!({
        "games_per_configuration": 10000,
        "total_games": total_games,
        "total_configurations": total_configurations,
        "avg_options_overall": avg_options_overall,
        "total_decisions_estimated": total_games as f64 * avg_options_overall,
        "suit_statistics": game_stats.iter().map(|(suits, total_games, deck, opts, min, max)| {
            serde_json::json!({
                "suit_count": suits,
                "total_games": total_games,
                "configurations": total_games / 10000,
                "games_per_configuration": 10000,
                "avg_deck_size": deck,
                "avg_options_per_turn": opts,
                "min_options": min,
                "max_options": max,
                "estimated_decisions_per_10k_games": deck * 0.75 * opts * 10000.0
            })
        }).collect::<Vec<_>>()
    });

    let json = serde_json::to_string_pretty(&stats_summary).unwrap();
    std::fs::write("game_statistics_summary.json", json)
        .expect("Failed to write game statistics to file");
    println!("\nGame statistics saved to 'game_statistics_summary.json'");
}

// Generate all combinations of strategies with repetition allowed
fn generate_strategy_combinations(player_count: usize, strategy_count: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = vec![0; player_count];

    generate_combinations_recursive(strategy_count, &mut current, 0, player_count, &mut result);

    result
}

fn generate_combinations_recursive(
    strategy_count: usize,
    current: &mut Vec<usize>,
    position: usize,
    player_count: usize,
    result: &mut Vec<Vec<usize>>,
) {
    if position == player_count {
        result.push(current.clone());
        return;
    }

    // Start from the last strategy used (or 0 if first position)
    // This ensures we only generate non-decreasing sequences (combinations)
    let start = if position == 0 {
        0
    } else {
        current[position - 1]
    };

    for strategy_index in start..strategy_count {
        current[position] = strategy_index;
        generate_combinations_recursive(
            strategy_count,
            current,
            position + 1,
            player_count,
            result,
        );
    }
}
