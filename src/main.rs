use std::fs::File;

use nanorand::{Rng, WyRand};
use once_cell::sync::Lazy;

use crate::{
    ds::{apply_action, Action, Configuration, Move, Phase, Pos, State, POSITIONS},
    mcts::mcts,
};

static mut RNG_TEST: Lazy<WyRand> = Lazy::new(|| WyRand::new());

mod ds;
mod mcts;

fn is_corner(p: Pos) -> bool {
    let (_, i) = p;
    (i % 2) == 1
}

pub fn is_muhle(conf: &Configuration, p: Pos) -> bool {
    let (r, i) = p;
    let state = conf.get(p);
    if is_corner(p) {
        (conf.get((r, (i + 1) % 8)) == state && conf.get((r, (i + 2) % 8)) == state)
            || (conf.get((r, (i + 8 - 2) % 8)) == state && conf.get((r, (i + 8 - 1) % 8)) == state)
    } else {
        (conf.get((r, (i + 1) % 8)) == state && conf.get((r, (i + 8 - 1) % 8)) == state)
            || (conf.get((0, i)) == state && conf.get((1, i)) == state && conf.get((2, i)) == state)
    }
}

fn has_pieces_not_in_muhle(conf: &Configuration, color: State) -> bool {
    POSITIONS.iter().copied().filter(|&p| conf.get(p) == color).any(|p| !is_muhle(&conf, p))
}

pub fn count(conf: &Configuration, color: State) -> usize {
    POSITIONS.iter().copied().filter(|&p| conf.get(p) == color).count()
}

fn for_each_neighbour<F>(p: Pos, mut func: F)
where
    F: FnMut(Pos),
{
    let (r, i) = p;
    func((r, (i + 1) % 8));
    func((r, (i + 8 - 1) % 8));
    if !is_corner(p) {
        if r != 2 {
            func((r + 1, i));
        }
        if r != 0 {
            func((r - 1, i));
        }
    }
}

fn for_each_reachable<F>(conf: &Configuration, p: Pos, mut func: F)
where
    F: FnMut(Pos),
{
    let can_jump = count(conf, conf.get(p)) == 3;
    if can_jump {
        POSITIONS.iter().copied().filter(|&p| conf.get(p) == State::Empty).for_each(func);
    } else {
        for_each_neighbour(p, |p| {
            if conf.get(p) == State::Empty {
                func(p);
            }
        });
    }
}

fn for_each_action<F>(conf: &Configuration, phase: Phase, color: State, mut func: F)
where
    F: FnMut(Action),
{
    match phase {
        Phase::Placement => {
            POSITIONS.iter().copied().filter(|&p| conf.get(p) == State::Empty).for_each(|p| {
                func(Action::Place(p));
            });
        }
        Phase::Moving => {
            POSITIONS.iter().copied().filter(|&p| conf.get(p) == color).for_each(|from| {
                for_each_reachable(conf, from, |to| {
                    func(Action::Move(from, to));
                })
            });
        }
    }
}

pub fn compute_actions(conf: &Configuration, phase: Phase, color: State) -> Vec<Action> {
    let mut res = Vec::new();
    for_each_action(conf, phase, color, |a| res.push(a));
    res
}

fn for_each_can_take<F>(conf: &Configuration, color: State, func: F)
where
    F: FnMut(Pos),
{
    let black_has_pieces_not_in_muhle = has_pieces_not_in_muhle(conf, color);
    POSITIONS
        .iter()
        .copied()
        .filter(|&p| conf.get(p) == color)
        .filter(|&p| !black_has_pieces_not_in_muhle || !is_muhle(conf, p))
        .for_each(func);
}

pub fn compute_can_take(conf: &Configuration, color: State) -> Vec<Pos> {
    let mut res = Vec::new();
    for_each_can_take(conf, color, |p| res.push(p));
    res
}

fn for_each_move<F>(conf: &Configuration, phase: Phase, color: State, mut func: F)
where
    F: FnMut(Move),
{
    for_each_action(conf, phase, color, |a| {
        let successor = apply_action(conf, a, color);
        let can_take = is_muhle(
            &successor,
            match a {
                Action::Place(p) => p,
                Action::Move(_from, to) => to,
            },
        );
        if can_take {
            for_each_can_take(conf, color.flip_color(), |take| {
                func(Move { action: a, take: Some(take) });
            });
        } else {
            func(Move { action: a, take: None });
        }
    });
}

pub fn compute_moves(conf: &Configuration, phase: Phase, color: State) -> Vec<Move> {
    let mut res = Vec::new();
    for_each_move(conf, phase, color, |m| res.push(m));
    res
}

fn main() {
     let arr = [
        [
            State::Empty,
            State::Black,
            State::Empty,
            State::White,
            State::Empty,
            State::Empty,
            State::Empty,
            State::White,
        ],
        [
            State::Empty,
            State::Empty,
            State::Black,
            State::Empty,
            State::Empty,
            State::Black,
            State::Black,
            State::Empty,
        ],
        [
            State::White,
            State::Empty,
            State::Empty,
            State::White,
            State::White,
            State::Empty,
            State::Black,
            State::Empty,
        ],
    ];

    mcts(&Configuration { arr }, 10, State::White);
    return;

    let mut total_moves_made = 0;
    let file = File::open("moves-neccessary_for_win.log");

    loop {
        let mut input = String::new();

        std::io::stdin().read_line(&mut input).expect("Failed to read line");

        let mut input = input.trim().split(" ");

        let phase = match input.next().unwrap() {
            "P" => Phase::Placement,
            "M" => Phase::Moving,
            _ => panic!("Unknown phase"),
        };

        let color = match input.next().unwrap() {
            "B" => State::Black,
            "W" => State::White,
            _ => panic!("Unknown color"),
        };

        let conf: Configuration = input.next().unwrap().parse().unwrap();

        // If there is already some stone on the playfield when this ai starts processing, another ai/ player did a move already
        if total_moves_made == 0 && (count(&conf, State::White) > 0 || count(&conf, State::Black) > 0) {
            total_moves_made += 1;
        }

        let selected = mcts(&conf, total_moves_made, color);

        /*         let moves = compute_moves(&conf, phase, color);
        let selectedd_move = mcts(&conf, total_moves_made, color);

        let i = unsafe { RNG_TEST.generate_range(0..moves.len()) };
        let selected = moves[i]; */
        println!("{}", selected.to_string());

        total_moves_made += 2;
    }
}
