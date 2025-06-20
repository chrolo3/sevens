use crate::cards::{Board, Card, GameBoard, GameRules, Hand, Strategy};
use rand::prelude::ThreadRng;
use rand::Rng;
use std::fmt;

pub struct Sevens;

impl GameRules for Sevens {
    fn can_play(&self, board: &Board, card: &Card) -> bool {
        // Early return for the most common case
        if card.value == board.n7 {
            return true;
        }

        let range = board.ranges[card.suit.rank];
        let card_value = card.value;

        // Simplified logic without pattern matching overhead
        (range.0 > 0 && range.0 - 1 == card_value) || (range.1 > 0 && range.1 + 1 == card_value)
    }

    fn find_options(&self, board: &Board) -> usize {
        let max_suit_range = 2 * board.n7 - 1;

        board
            .ranges
            .iter()
            .map(|&range| {
                match range {
                    (0, 0) => 1,                        // Empty range: 1 option
                    (1, r) if r == max_suit_range => 0, // Full range: 0 options
                    (1, _) => 1,                        // Min reached: 1 option
                    (_, r) if r == max_suit_range => 1, // Max reached: 1 option
                    _ => 2,                             // Normal case: 2 options
                }
            })
            .sum()
    }
}

#[derive(Clone)]
pub struct VanillaRandom;

impl Strategy for VanillaRandom {
    type Rules = Sevens;

    fn select_card(
        &self,
        hand: &Hand,
        board: &GameBoard<Self::Rules>,
        rng: &mut ThreadRng,
    ) -> Option<usize> {
        let mut choices: Vec<usize> = Vec::with_capacity(hand.len());

        // First pass: find the 7 of Spades and all playable cards
        for (index, card) in hand.iter().enumerate() {
            if board.can_play(card) {
                choices.push(index);
            }
        }

        // Otherwise, play a random card from the valid choices
        if !choices.is_empty() {
            Some(choices[rng.random_range(0..choices.len())]) // Use gen_range for uniform distribution
        } else {
            None
        }
    }
}

impl fmt::Debug for VanillaRandom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "A")
    }
}

#[derive(Clone)]
pub struct LowestFirst;
impl Strategy for LowestFirst {
    type Rules = Sevens;

    fn select_card(
        &self,
        hand: &Hand,
        board: &GameBoard<Self::Rules>,
        rng: &mut ThreadRng,
    ) -> Option<usize> {
        let mut choices: Vec<usize> = Vec::with_capacity(hand.len());

        // First pass: find the 7 of Spades and all playable cards
        for (index, card) in hand.iter().enumerate() {
            if board.can_play(card) {
                choices.push(index);
            }
        }

        if !choices.is_empty() {
            let mut lowest_suit_card: Option<Card> = None;
            let mut lowest_index: Option<usize> = None;

            // Find the highest non-spade card
            for index in choices.iter() {
                if let Some(card) = lowest_suit_card.clone() {
                    if hand[*index].value < card.value {
                        lowest_suit_card = Some(hand[*index].clone());
                        lowest_index = Some(*index);
                    }
                } else {
                    lowest_suit_card = Some(hand[*index].clone());
                    lowest_index = Some(*index);
                }
            }

            if let Some(index) = lowest_index {
                return Some(index);
            }

            Some(choices[rng.random_range(0..choices.len())])
        } else {
            None
        }
    }
}
impl fmt::Debug for LowestFirst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B")
    }
}

#[derive(Clone)]
pub struct HighestFirst;
impl Strategy for HighestFirst {
    type Rules = Sevens;

    fn select_card(
        &self,
        hand: &Hand,
        board: &GameBoard<Self::Rules>,
        rng: &mut ThreadRng,
    ) -> Option<usize> {
        let mut choices: Vec<usize> = Vec::with_capacity(hand.len());

        // First pass: find the 7 of Spades and all playable cards
        for (index, card) in hand.iter().enumerate() {
            if board.can_play(card) {
                choices.push(index);
            }
        }

        if !choices.is_empty() {
            let mut highest_suit_card: Option<Card> = None;
            let mut highest_index: Option<usize> = None;

            // Find the highest non-spade card
            for index in choices.iter() {
                if let Some(card) = highest_suit_card.clone() {
                    if hand[*index].value > card.value {
                        highest_suit_card = Some(hand[*index].clone());
                        highest_index = Some(*index);
                    }
                } else {
                    highest_suit_card = Some(hand[*index].clone());
                    highest_index = Some(*index);
                }
            }

            if let Some(index) = highest_index {
                return Some(index);
            }

            Some(choices[rng.random_range(0..choices.len())])
        } else {
            None
        }
    }
}
impl fmt::Debug for HighestFirst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C")
    }
}
