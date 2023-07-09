use std::str::FromStr;

pub const SIDES: usize = 4;
pub const SIDESXTWO: usize = SIDES * 2;

pub type Pos = (u8, u8);

pub const POSITIONS: [Pos; 3 * SIDESXTWO] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
];

#[derive(Copy, Clone, PartialEq)]
pub enum Phase {
    Placement,
    Moving,
}

#[derive(Copy, Clone, PartialEq)]
pub enum Action {
    Place(Pos),
    Move(Pos, Pos),
}

pub fn apply_action(conf: &Configuration, action: Action, color: State) -> Configuration {
    let mut conf = conf.clone();
    match action {
        Action::Place(p) => {
            conf.set(p, color);
        }
        Action::Move(from, to) => {
            conf.set(to, conf.get(from));
            conf.set(from, State::Empty);
        }
    }
    conf
}

#[derive(Copy, Clone, PartialEq)]
pub struct Move {
    pub(crate) action: Action,
    pub(crate) take: Option<Pos>,
}

fn pos_to_string(p: Pos) -> String {
    format!("{}", POSITIONS.iter().position(|&p1| p == p1).unwrap() + 1)
}

impl ToString for Move {
    fn to_string(&self) -> String {
        format!("{}{}",
        match self.action {
            Action::Place(t) => { format!("P {}", pos_to_string(t)) }
            Action::Move(s, t) => { format!("M {} {}", pos_to_string(s), pos_to_string(t))}
        },
        match self.take {
            None => { String::new() }
            Some(t) => { format!(" T {}", pos_to_string(t)) }
        }
        )
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Default, Debug)]
pub enum State {
    #[default]
    Empty,
    White,
    Black,
}

impl State {
    pub fn flip_color(&self) -> State {
        match self {
            State::Empty => {
                State::Empty
            }
            State::White => {
                State::Black
            }
            State::Black => {
                State::White
            }
        }
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Default, Debug)]
pub struct Configuration {
    arr: [[State; SIDESXTWO]; 3],
}

impl Configuration {
    pub(crate) fn get(&self, p: Pos) -> State {
        let (r, i) = p;
        self.arr[r as usize][i as usize]
    }

    pub(crate) fn set(&mut self, p: Pos, state: State) {
        let (r, i) = p;
        self.arr[r as usize][i as usize] = state;
    }
}

impl FromStr for Configuration {
    type Err = ();

    fn from_str(line: &str) -> Result<Self, Self::Err> {
        let mut conf = Configuration::default();
        let mut iter = line.chars();
        for p in POSITIONS {
            match iter.next().unwrap() {
                'W' => {
                    conf.set(p, State::White);
                }
                'B' => {
                    conf.set(p, State::Black);
                }
                'E' => {
                    conf.set(p, State::Empty);
                }
                _ => {
                    return Err(());
                }
            }
        }
        Ok(conf)
    }
}