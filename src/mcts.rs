use std::ops::Not;

use id_tree::{Node, NodeId, Tree, TreeBuilder};
use nanorand::{Rng, WyRand};
use once_cell::sync::Lazy;

use crate::{
    compute_moves, count,
    ds::{apply_action, Configuration, Move, Phase, State},
    for_each_move,
};

pub static mut RNG: Lazy<WyRand> = Lazy::new(|| WyRand::new());

const TREE_INIT_NODE_COUNT: usize = 2_000_000;

#[derive(PartialEq)]
enum Nash {
    WON,
    DRAW,
    LOST,
}

impl Not for Nash {
    type Output = Nash;

    fn not(self) -> Self::Output {
        match self {
            Nash::WON => Nash::LOST,
            Nash::DRAW => Nash::DRAW,
            Nash::LOST => Nash::WON,
        }
    }
}

impl From<u32> for Nash {
    fn from(value: u32) -> Self {
        match value {
            1 => Nash::WON,
            0 | u32::MAX => {
                panic!("Tried to covert 0 or overflown -1 to Nash, which doesn't make sense for a-b-pruning")
            }
            _ => panic!("Tried to convert some unsigned integer to Nash, which makes no sens :)"),
        }
    }
}

impl From<u32> for Phase {
    fn from(value: u32) -> Self {
        if (value / 2) < 9 {
            Phase::Placement
        } else {
            Phase::Moving
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
struct MCTSNodeContent {
    config: Configuration,
    move_count: u32,
    current_turn: State, // = player_color
    visit_count: u32,    // v_i
    win_count: u32,      // w_i
}

impl MCTSNodeContent {
    fn new(config: Configuration, move_count: u32, current_turn: State) -> MCTSNodeContent {
        MCTSNodeContent {
            config,
            move_count,
            current_turn,
            visit_count: 0,
            win_count: u32::MAX,
        }
    }

    /// Computes the value which is used to choose the best path to walk down the tree in the selection-phase
    fn calculate_uct(&self, parrent_node_visit_ount: f64) -> f64 {
        return if self.win_count == u32::MAX {
            f64::INFINITY
        } else if self.win_count == 0 {
            0f64
        } else {
            (self.win_count as f64 / self.visit_count as f64)
                + f64::sqrt(f64::ln(parrent_node_visit_ount) / self.visit_count as f64)
        };
    }
}

/// Calculates Best Move for current Config
pub fn mcts(config: &Configuration, move_count: u32, current_turn: State) -> Configuration {
    let root_node_content = MCTSNodeContent::new(*config, move_count, current_turn);

    let mut tree = TreeBuilder::<MCTSNodeContent>::new()
        .with_node_capacity(TREE_INIT_NODE_COUNT)
        .with_root(Node::new(root_node_content))
        .build();

    let mut buff = Vec::<Move>::new();

    // As long as time is available, do:
    loop {
        let node_max_uct = selction(&tree);
        let nodes_expansion_id = expansion(&mut tree, node_max_uct);

        let nodes_color = tree.get(&nodes_expansion_id).unwrap().data().current_turn;
        let nodes_move_count = tree.get(&nodes_expansion_id).unwrap().data().move_count;
        let nash = simulation(*config, nodes_color, nodes_move_count, current_turn, &mut buff);
    }
}

fn apply_move(config: &Configuration, m: &Move, start_color: State) -> Configuration {
    let mut modified_config = apply_action(config, m.action, start_color);
    if let Some(to_take) = m.take {
        modified_config.set(to_take, State::Empty);
    }
    modified_config
}

// TODO How to determine this?!
fn get_phase(config: &Configuration) -> Option<Phase> {
    let (white_count, black_count) = (count(config, State::White), count(config, State::Black));

    Option::None
}

/// Selects ChildNode that hasn't been visited, which equals the leftmost child node where calculate_uct is `f64::INFINITY`
fn selction(tree: &Tree<MCTSNodeContent>) -> (NodeId, MCTSNodeContent) {
    let mut current_node_id = tree.root_node_id().unwrap();
    let mut current_node = tree.get(current_node_id).unwrap();

    // Search the tree along the path with maximal UCT value until node is reached, which not was visited so far
    while current_node.data().visit_count != 0 {
        let child_ids = tree.children_ids(current_node_id).unwrap();

        // Selects a node with the highest UTC-value from the current nodes child nodes
        let next_node_id = child_ids
            .max_by(|id1, id2| {
                let currents_visit_count = current_node.data().visit_count as f64;

                let node_1_uct = tree.get(id1).unwrap().data().calculate_uct(currents_visit_count);
                let node_2_uct = tree.get(id2).unwrap().data().calculate_uct(currents_visit_count);

                node_1_uct.total_cmp(&node_2_uct)
            })
            .unwrap();

        // Move to the next niveau
        current_node_id = next_node_id;
        current_node = tree.get(current_node_id).unwrap();
    }

    (current_node_id.clone(), *current_node.data())
}

/// Adds (all 'suitable') child nodes of the givin node to the tree and returns NodeId from one of them randomly
/// In Application: the selected Node.
fn expansion(tree: &mut Tree<MCTSNodeContent>, leaf: (NodeId, MCTSNodeContent)) -> NodeId {
    use id_tree::InsertBehavior::*;

    let (leaf_id, leaf_node) = leaf;
    let current_move_color = leaf_node.current_turn;
    let child_move_color = current_move_color.flip_color();

    let moves = compute_moves(&leaf_node.config, Phase::from(leaf_node.move_count), current_move_color);
    let child_node_ids: Vec<NodeId> = moves
        .iter()
        .map(|m| apply_move(&leaf_node.config, m, current_move_color))
        .map(|config| {
            tree.insert(
                Node::new(MCTSNodeContent::new(config, leaf_node.move_count + 1, child_move_color)),
                UnderNode(&leaf_id),
            )
            .unwrap()
        })
        .collect();

    // TODO use some evaluation function right here
    let i = unsafe { RNG.generate_range(0..child_node_ids.len()) };
    child_node_ids[i].clone()
}

///
fn simulation(
    leaf_config: Configuration,
    leaf_color: State,
    mut move_count: u32,
    root_color: State,
    buff: &mut Vec<Move>,
) -> Nash {
    let mut current_config = leaf_config;
    let mut current_color = leaf_color;
    let mut nash = Nash::DRAW;

    // TODO break the function when the path is too long and there is a high possibility it is a draw
    // While not-terminal-config do:
    while nash == Nash::DRAW {
        apply_rand_move(&mut current_config, Phase::Moving, current_color, buff);
        move_count += 1;
        buff.clear();
        current_color = current_color.flip_color();

        calculate_nash_for_leaf(&current_config, current_color, leaf_color, Phase::from(move_count), &mut nash);

        /* let moves = compute_moves(&current_config, Phase::Moving /* TODO */, current_color);
        let i = unsafe { RNG.generate_range(0..moves.len()) };
        current_config = apply_move(&current_config, &moves[i], color); */

        /* if is_terminal(current_config) {break calculate_nash(current_config, root_start);} */
    }
    nash
}

/// Computes possible moves from config, chooses one randomly, updates config with new one
// TODO: Evalutation-Function instead of RNG
fn apply_rand_move(conf: &mut Configuration, phase: Phase, color: State, buff: &mut Vec<Move>) {
    for_each_move(conf, phase, color, |m| buff.push(m));
    let i = unsafe { RNG.generate_range(0..buff.len()) };
    *conf = apply_move(conf, &buff[i], color);
}

/// Calculates back probagated nash value for leaf color and updates it
fn calculate_nash_for_leaf(
    conf: &Configuration,
    current_color: State,
    leaf_color: State,
    phase: Phase,
    nash: &mut Nash,
) {
    //nash is DRAW by default & stays like that during Placement-Phase due to the game being not decidable yet
    if phase == Phase::Placement {
        return;
    }

    if count(conf, current_color) == 2 {
        if current_color == leaf_color {
            *nash = Nash::LOST;
        } else {
            *nash = Nash::WON;
        }
    }

    if compute_moves(conf, Phase::Moving, current_color).is_empty() {
        if current_color == leaf_color {
            *nash = Nash::LOST;
        } else {
            *nash = Nash::WON;
        }
    }
}

fn back_propagation(tree: &mut Tree<MCTSNodeContent>, leaf_id: NodeId, nash: i32, leaf_turn: State) {
    let mut current_node_id = leaf_id.clone();
    let mut current_node = tree.get_mut(&leaf_id).unwrap();
    let mut current_turn = leaf_turn;

    loop {
        current_node.data_mut().visit_count += 1;
        if (nash == 1 && current_turn == leaf_turn) || (nash == -1 && current_turn == leaf_turn.flip_color()) {
            current_node.data_mut().win_count += 1;
        }

        (current_node_id, current_node) = if let Ok(ancenstor_ids) = tree.ancestor_ids(&leaf_id) {
            let parent_id = ancenstor_ids.last().unwrap();
            (parent_id.clone(), tree.get_mut(&parent_id.clone()).unwrap())
        } else {
            break;
        };
        current_turn = current_turn.flip_color();
    }
}
