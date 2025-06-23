use crate::cards::{Board, Card, GameBoard, GameRules, Hand, Strategy};
use rand::prelude::ThreadRng;
use rand::Rng;
use std::fmt;

pub struct SevensSpades;

impl GameRules for SevensSpades {
    fn can_play(&self, board: &Board, card: &Card) -> bool {
        let range = board.ranges[card.suit.rank];
        let card_value = card.value;

        // Early return for 7s (most common playable card)
        if card_value == board.n7 {
            return true;
        }

        // Check if card can be placed at the edge of its own suit's range
        let can_extend_own_suit = (range.0 > 0 && range.0 - 1 == card_value)
            || (range.1 > 0 && range.1 + 1 == card_value);

        if card.suit.rank == 0 {
            // Spades can always be played if they extend the range
            can_extend_own_suit
        } else {
            // Non-spades need to check if they're in the spade range first
            let spade_range = board.ranges[0];
            let in_spade_range = spade_range.0 > 0
                && spade_range.1 > 0
                && card_value >= spade_range.0
                && card_value <= spade_range.1;

            in_spade_range && can_extend_own_suit
        }
    }

    fn find_options(&self, board: &Board) -> usize {
        let max_suit_range = 2 * board.n7 - 1;
        let spades_range = board.ranges[0];
        let n7 = board.n7;
        // Pre-compute spade range check for 7s
        let can_play_seven = spades_range.0 <= n7 && n7 <= spades_range.1;
        board
            .ranges
            .iter()
            .enumerate()
            .map(|(suit_index, &range)| {
                match range {
                    (0, 0) => {
                        if suit_index == 0 {
                            // Spades suit - can always be started
                            1
                        } else {
                            // Other unplayed suits - need 7 to be playable
                            if can_play_seven {
                                1
                            } else {
                                0
                            }
                        }
                    }
                    (low, high) => {
                        // Spades can always be extended (unless full range)
                        if suit_index == 0 {
                            match (low, high) {
                                (1, h) if h == max_suit_range => 0, // Full range
                                (1, _) => 1,                        // Min reached
                                (_, h) if h == max_suit_range => 1, // Max reached
                                _ => 2,                             // Normal case
                            }
                        } else {
                            // Non-spades need to check spade range constraints
                            // Check if range is entirely within spades or touches boundaries
                            let within_spades = low > spades_range.0 && high < spades_range.1;
                            let is_empty_or_within = range == (0, 0) || within_spades;
                            if is_empty_or_within {
                                // Standard logic for suits within spade range
                                match (low, high) {
                                    (1, h) if h == max_suit_range => 0, // Full range
                                    (1, _) => 1,                        // Min reached
                                    (_, h) if h == max_suit_range => 1, // Max reached
                                    _ => 2,                             // Normal case
                                }
                            } else {
                                // Count valid extensions within spade range
                                let mut options = 0;
                                // Check lower extension
                                if low > 1 {
                                    let next_low = low - 1;
                                    if next_low >= spades_range.0 && next_low <= spades_range.1 {
                                        options += 1;
                                    }
                                }
                                // Check upper extension
                                if high < max_suit_range {
                                    let next_high = high + 1;
                                    if next_high >= spades_range.0 && next_high <= spades_range.1 {
                                        options += 1;
                                    }
                                }
                                options
                            }
                        }
                    }
                }
            })
            .sum()
    }
}

#[derive(Clone)]
pub struct SpadesRandom;

impl Strategy for SpadesRandom {
    type Rules = SevensSpades;

    fn select_card(
        &self,
        hand: &Hand,
        board: &GameBoard<Self::Rules>,
        rng: &mut ThreadRng,
    ) -> Option<usize> {
        let mut choices: Vec<usize> = Vec::new();
        let mut seven_of_spades_index: Option<usize> = None;
        let spade_rank = 0;

        // First pass: find the 7 of Spades and all playable cards
        for (index, card) in hand.iter().enumerate() {
            if card.suit.rank == spade_rank && card.value == board.board.n7 {
                seven_of_spades_index = Some(index);
            }
            if board.can_play(card) {
                choices.push(index);
            }
        }

        // Always play the 7 of Spades if we have it
        if let Some(index) = seven_of_spades_index {
            return Some(index);
        }

        // Otherwise, play a random card from the valid choices
        if !choices.is_empty() {
            Some(choices[rng.random_range(0..choices.len())])
        } else {
            None
        }
    }
}

impl fmt::Debug for SpadesRandom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "A")
    }
}

#[derive(Clone)]
pub struct SpadeFirstStrategy;
impl Strategy for SpadeFirstStrategy {
    type Rules = SevensSpades;

    fn select_card(
        &self,
        hand: &Hand,
        board: &GameBoard<Self::Rules>,
        rng: &mut ThreadRng,
    ) -> Option<usize> {
        let mut choices: Vec<usize> = Vec::new();
        let mut seven_of_spades_index: Option<usize> = None;
        let spade_rank = 0;

        // First pass: find the 7 of Spades and all playable cards
        for (index, card) in hand.iter().enumerate() {
            if card.suit.rank == spade_rank && card.value == board.board.n7 {
                seven_of_spades_index = Some(index);
            }
            if board.can_play(card) {
                choices.push(index);
            }
        }

        // Always play the 7 of Spades if we have it
        if let Some(index) = seven_of_spades_index {
            return Some(index);
        }

        if !choices.is_empty() {
            // Prefer spades when available
            for index in choices.iter() {
                if hand[*index].suit.rank == spade_rank {
                    return Some(*index);
                }
            }
            // Otherwise random
            Some(choices[rng.random_range(0..choices.len())])
        } else {
            None
        }
    }
}
impl fmt::Debug for SpadeFirstStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B")
    }
}

#[derive(Clone)]
pub struct SpadeLastRandom;
impl Strategy for SpadeLastRandom {
    type Rules = SevensSpades;

    fn select_card(
        &self,
        hand: &Hand,
        board: &GameBoard<Self::Rules>,
        rng: &mut ThreadRng,
    ) -> Option<usize> {
        let mut choices: Vec<usize> = Vec::new();
        let mut seven_of_spades_index: Option<usize> = None;
        let spade_rank = 0;

        // First pass: find the 7 of Spades and all playable cards
        for (index, card) in hand.iter().enumerate() {
            if card.suit.rank == spade_rank && card.value == board.board.n7 {
                seven_of_spades_index = Some(index);
            }
            if board.can_play(card) {
                choices.push(index);
            }
        }

        // Always play the 7 of Spades if we have it
        if let Some(index) = seven_of_spades_index {
            return Some(index);
        }

        if !choices.is_empty() {
            // Prefer non-spades when available
            for index in choices.iter() {
                if hand[*index].suit.rank != spade_rank {
                    return Some(*index);
                }
            }
            // Otherwise random
            Some(choices[rng.random_range(0..choices.len())])
        } else {
            None
        }
    }
}
impl fmt::Debug for SpadeLastRandom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C")
    }
}

#[derive(Clone)]
pub struct SpadesLastHighest;
impl Strategy for SpadesLastHighest {
    type Rules = SevensSpades;

    fn select_card(
        &self,
        hand: &Hand,
        board: &GameBoard<Self::Rules>,
        rng: &mut ThreadRng,
    ) -> Option<usize> {
        let mut choices: Vec<usize> = Vec::new();
        let mut seven_of_spades_index: Option<usize> = None;
        let spade_rank = 0;

        // First pass: find the 7 of Spades and all playable cards
        for (index, card) in hand.iter().enumerate() {
            if card.suit.rank == spade_rank && card.value == board.board.n7 {
                seven_of_spades_index = Some(index);
            }
            if board.can_play(card) {
                choices.push(index);
            }
        }

        // Always play the 7 of Spades if we have it
        if let Some(index) = seven_of_spades_index {
            return Some(index);
        }

        if !choices.is_empty() {
            let mut highest_suit_card: Option<Card> = None;
            let mut highest_index: Option<usize> = None;

            // Find the highest non-spade card
            for index in choices.iter() {
                if hand[*index].suit.rank != spade_rank {
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
impl fmt::Debug for SpadesLastHighest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "D")
    }
}
