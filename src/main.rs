use mill_playfield::{EfficientPlayField, FieldPos, FieldState};
use nanorand::{WyRand, Rng};
use once_cell::sync::Lazy;

use crate::ds::{Action, apply_action, Move, Phase, POSITIONS};

mod ds;
mod mcts;

static mut RNG: Lazy<WyRand> = Lazy::new(|| WyRand::default());

fn is_corner(p: FieldPos) -> bool {
    (p.index % 2) == 1
}

pub fn is_muhle(conf: &EfficientPlayField, p: FieldPos) -> bool {
    let FieldPos{ring_index, index: i} = p;
    let state = conf.get_field_state(p);

    if is_corner(p) {
        (conf.get_field_state(FieldPos{ring_index, index: (i + 1) % 8}) == state
            && conf.get_field_state(FieldPos{ring_index, index: (i + 2) % 8}) == state)
        || (conf.get_field_state(FieldPos{ring_index, index: (i + 8 - 2) % 8}) == state
            && conf.get_field_state(FieldPos { ring_index, index: (i + 8 - 1) % 8}) == state)
    } else {
        (conf.get_field_state(FieldPos{ring_index, index: (i + 1) % 8}) == state
            && conf.get_field_state(FieldPos{ring_index, index: (i + 8 - 1) % 8}) == state)
        || (conf.get_field_state(FieldPos{ring_index: 0, index: i}) == state
            && conf.get_field_state(FieldPos{ring_index: 1, index: i}) == state
            && conf.get_field_state(FieldPos{ ring_index: 2, index: i}) == state)
    }
    /*
    let FieldPos{ring_index: _r, index: i} = p;
    let state = conf.get_field_state(p);

    let rep_indices_to_rotate = (((p.index - (p.index % 2) + 7) % 8) * 2) as u32;
    // Field state triple containing field_index:
    let state_triple = conf.state[p.ring_index].rotate_right(rep_indices_to_rotate) & 0b0000_0000_0011_1111u16;

    /* 010101 | 101010 */
    if state_triple == 21u16 || state_triple == 42u16 {
        return true;
    }

    // If index is located in an edge, two triples must be checked for mill occurrence
    return if is_corner(p) {
        let state_triple = conf.state[p.ring_index].rotate_right(p.index * 2) & 0b0000_0000_0011_1111u16;
        (state == FieldState::White && state_triple == 21u16) || (state == FieldState::Black && state_triple == 42u16)
    } else {
        conf.get_field_state(FieldPos{ring_index: 0, index: i}) == conf.get_field_state(FieldPos{ring_index: 1, index: i})
        && conf.get_field_state(FieldPos{ ring_index: 1, index: i}) == conf.get_field_state(FieldPos{ ring_index: 2, index: i})
        && conf.get_field_state(FieldPos{ ring_index: 0, index: i}) == state
    }
     */
}

fn has_pieces_not_in_muhle(conf: &EfficientPlayField, color: FieldState) -> bool {
    POSITIONS.iter().copied().filter(|&p| conf.get_field_state(p) == color).any(|p| !is_muhle(&conf, p))
}

pub fn count_stones_of_color(conf: &EfficientPlayField, color: FieldState) -> usize {
    POSITIONS.iter().copied().filter(|&p| conf.get_field_state(p) == color).count()
}

fn for_each_neighbour<F>(p: FieldPos, mut func: F)
    where F: FnMut(FieldPos) {
    let FieldPos{ring_index: r, index: i} = p;
    func(FieldPos{index: (i + 1) % 8, ..p});
    func(FieldPos { index: (i + 8 - 1) % 8,..p});
    if !is_corner(p) {
        if r != 2 {
            func(FieldPos{ring_index: r + 1, ..p});
        }
        if r != 0 {
            func(FieldPos{ring_index: r - 1, ..p});
        }
    }
}

fn for_each_reachable_field<F>(conf: &EfficientPlayField, p: FieldPos, mut func: F)
    where F: FnMut(FieldPos) {
    let can_jump = count_stones_of_color(conf, conf.get_field_state(p)) == 3;
    if can_jump {
        POSITIONS.iter().copied().filter(|&p| conf.get_field_state(p) == FieldState::Free).for_each(func);
    } else {
        for_each_neighbour(p, |p| {
            if conf.get_field_state(p) == FieldState::Free {
                func(p);
            }
        });
    }
}

fn for_each_action<F>(conf: &EfficientPlayField, phase: Phase, color: FieldState, mut func: F)
    where F: FnMut(Action) {
    match phase {
        Phase::Placement => {
            POSITIONS.iter().copied().filter(|&p| conf.get_field_state(p) == FieldState::Free).for_each(|p| {
                func(Action::Place(p));
            });
        }
        Phase::Moving => {
            POSITIONS.iter().copied().filter(|&p| conf.get_field_state(p) == color).for_each(|from| {
                for_each_reachable_field(conf, from, |to| {
                    func(Action::Move(from, to));
                })
            });
        }
    }
}

pub fn compute_actions(conf: &EfficientPlayField, phase: Phase, color: FieldState) -> Vec<Action> {
    let mut res = Vec::new();
    for_each_action(conf, phase, color, |a| res.push(a));
    res
}

fn for_each_can_take<F>(conf: &EfficientPlayField, color: FieldState, func: F)
    where F: FnMut(FieldPos) {
    let black_has_takable_pieces = has_pieces_not_in_muhle(conf, /* ! */color); // TODO
    POSITIONS.iter().copied().filter(|&p| conf.get_field_state(p) == color)
        .filter(|&p| !black_has_takable_pieces || !is_muhle(conf, p))
        .for_each(func);
}

pub fn compute_can_take(conf: &EfficientPlayField, color: FieldState) -> Vec<FieldPos> {
    let mut res = Vec::new();
    for_each_can_take(conf, color, |p| res.push(p));
    res
}

fn for_each_move<F>(conf: &EfficientPlayField, phase: Phase, color: FieldState, mut func: F)
    where F: FnMut(Move) {
    for_each_action(conf, phase, color, |a| {
        let successor = apply_action(conf, a, !color);
        let can_take = is_muhle(&successor, match a {
            Action::Place(p) => { p }
            Action::Move(_from, to) => { to }
        });
        if can_take {
            for_each_can_take(conf, !color, |take| { // TODO not &successor?
                func(Move {
                    action: a,
                    take: Some(take),
                });
            });
        } else {
            func(Move {
                action: a,
                take: None,
            });
        }
    });
}

pub fn compute_moves(conf: &EfficientPlayField, phase: Phase, color: FieldState) -> Vec<Move> {
    let mut res = Vec::new();
    for_each_move(conf, phase, color, |m| res.push(m));
    res
}


fn main() {
    loop {
        let mut input = String::new();

        std::io::stdin().read_line(&mut input).expect("Failed to read line");

        let mut input = input.trim().split(" ");

        let phase = match input.next().unwrap() {
            "P" => Phase::Placement,
            "M" => Phase::Moving,
            _ => panic!("Unknown phase")
        };

        let color = match input.next().unwrap() {
            "B" => FieldState::Black,
            "W" => FieldState::White,
            _ => panic!("Unknown color")
        };

        let conf = EfficientPlayField::from_coded(input.next().unwrap());

        let moves = compute_moves(&conf, phase, color);
        let i = unsafe {
           RNG.generate_range(0..moves.len())
        };
        let selected = moves[i];

        eprintln!("{conf}\n---");
        for m in moves {
            eprintln!("{}", m.to_string())
        }

        println!("{}", selected.to_string());
    }
}
