use mill_playfield::{FieldPos, FieldState, EfficientPlayField};

pub const SIDES: usize = 4;
pub const SIDESXTWO: usize = SIDES * 2;

pub const POSITIONS: [FieldPos; 3 * SIDESXTWO] = [
    FieldPos{ ring_index: 2, index: 0}, FieldPos{ ring_index: 2, index: 1}, FieldPos{ ring_index: 2, index: 2}, FieldPos{ ring_index: 2, index: 3}, FieldPos{ ring_index: 2, index: 4}, FieldPos{ ring_index: 2, index: 5}, FieldPos{ ring_index: 2, index: 6}, FieldPos{ ring_index: 2, index: 7},
    FieldPos{ ring_index: 1, index: 0}, FieldPos{ ring_index: 1, index: 1}, FieldPos{ ring_index: 1, index: 2}, FieldPos{ ring_index: 1, index: 3}, FieldPos{ ring_index: 1, index: 4}, FieldPos{ ring_index: 1, index: 5}, FieldPos{ ring_index: 1, index: 6}, FieldPos{ ring_index: 1, index: 7},
    FieldPos{ ring_index: 0, index: 0}, FieldPos{ ring_index: 0, index: 1}, FieldPos{ ring_index: 0, index: 2}, FieldPos{ ring_index: 0, index: 3}, FieldPos{ ring_index: 0, index: 4}, FieldPos{ ring_index: 0, index: 5}, FieldPos{ ring_index: 0, index: 6}, FieldPos{ ring_index: 0, index: 7},
];

#[derive(Copy, Clone, PartialEq)]
pub enum Phase {
    Placement,
    Moving,
}

#[derive(Copy, Clone, PartialEq)]
pub enum Action {
    Place(FieldPos),
    Move(FieldPos, FieldPos),
}

#[derive(Copy, Clone, PartialEq)]
pub struct Move {
    pub(crate) action: Action,
    pub(crate) take: Option<FieldPos>,
}

impl ToString for Move {
    fn to_string(&self) -> String {
        format!("{}{}",
            match self.action {
                Action::Place(t) => { format!("P {}", pos_to_string(t)) }
                Action::Move(s, t) => { format!("M {} {}", pos_to_string(s), pos_to_string(t))}
            },
            &match self.take {
                None => { String::new() }
                Some(t) => { format!(" T {}", pos_to_string(t)) }
            }
        )
    }
}

pub fn apply_action(conf: &EfficientPlayField, action: Action, color: FieldState) -> EfficientPlayField {
    let mut conf = conf.clone();
    match action {
        Action::Place(p) => {
            conf.set_field_state(p, color);
        }
        Action::Move(from, to) => {
            conf.set_field_state(to, conf.get_field_state(from));
            conf.set_field_state(from, FieldState::Free);
        }
    }
    conf
}

fn pos_to_string(p: FieldPos) -> String {
    format!("{}", POSITIONS.iter().position(|&p1| p == p1).unwrap() + 1)
}