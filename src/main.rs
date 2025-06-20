mod cards;
mod spades;
mod vanilla;

fn main() { 
    cards::run_simulations_sevens(4, 10000);
    cards::run_simulations_spades(4, 10000);
}
