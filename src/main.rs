use crate::{
    ds::{apply_action, Action, Configuration, Move, Phase, Pos, State, POSITIONS},
    mcts::{mcts, MCTSNodeContent, TREE_INIT_NODE_COUNT},
};

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

fn main() -> ! {
    let mut total_moves_made = 0;

    // initilize tree for the continuous mcts pass on to avoid multiple calculations on parts of the same subtree
    let mut tree = id_tree::TreeBuilder::<MCTSNodeContent>::new().with_node_capacity(TREE_INIT_NODE_COUNT).build();

    let mut last_mcts_pick_node_id = Option::None;

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
        if phase != Phase::from(total_moves_made) {
            panic!("Programmer dumb lol");
        }

        let (selected_move, selected_node_id, mod_tree) =
            mcts(tree, &mut last_mcts_pick_node_id, &conf, total_moves_made, color);
        tree = mod_tree;
        last_mcts_pick_node_id = Option::Some(selected_node_id);

        println!("{}", selected_move.to_string());

        // Adding all possible opponent moves to the tree, when they aren't already present there
/*         let last_picked_children: Vec<Configuration> =
            tree.children(last_mcts_pick_node_id.as_ref().unwrap()).unwrap().map(|n| n.data().clone().conf).collect();

        let last_picked_node = tree.get(last_mcts_pick_node_id.as_ref().unwrap()).unwrap().data().clone();
        for_each_move(&last_picked_node.conf, Phase::from(total_moves_made + 1), color.flip_color(), |m| {
            let mod_conf = apply_move(&last_picked_node.conf, &m, color.flip_color());

            if !last_picked_children.contains(&mod_conf) {
                tree.insert(
                    Node::new(MCTSNodeContent::new(mod_conf, u32::MAX, 1)),
                    id_tree::InsertBehavior::UnderNode(last_mcts_pick_node_id.as_ref().unwrap()),
                )
                .unwrap();
            }
        }); */

        // Finally update the mode count
        total_moves_made += 2;
    }
}
