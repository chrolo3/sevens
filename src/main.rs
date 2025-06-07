mod cards;
mod spades;
mod vanilla;


fn main() { 
    cards::run_simulations(4, 10000);
}
