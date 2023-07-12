use std::{ops::Not, time::Instant};

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
    conf: Configuration,
    move_count: u32,     // to determine the game phase
    current_turn: State, // = player_color
    visit_count: u32,    // v_i
    win_count: u32,      // w_i
}

impl MCTSNodeContent {
    fn new(conf: Configuration, move_count: u32, current_turn: State) -> MCTSNodeContent {
        MCTSNodeContent {
            conf,
            move_count,
            current_turn,
            visit_count: 0,
            win_count: u32::MAX,
        }
    }

    /// Computes the value which is used to choose the best path to walk down the tree in the selection-phase
    /// Takes care of the extreme cases:
    /// - The node was not visited before => MAX is returned to use this node in the select-phase
    /// - The node was visited before, but the win_count == 0 => 0f64 is returned to avoid NaN
    ///   due to 0/v_i of the UTC formula
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
pub fn mcts(conf: &Configuration, move_count: u32, current_turn: State) -> Move {
    let instant = Instant::now();

    let root_node_content = MCTSNodeContent::new(*conf, move_count, current_turn);

    let mut tree = TreeBuilder::<MCTSNodeContent>::new()
        .with_node_capacity(TREE_INIT_NODE_COUNT)
        .with_root(Node::new(root_node_content))
        .build();

    let mut simulation_buffer = Vec::<Move>::with_capacity(200);

    while instant.elapsed().as_millis() < 1_000 {
        for _i in 0..10_000 {
            // Should be most left, non-visited leaf node on the path with the highest UCT value
            let unexplored_leaf_node = selection(&tree);

            // The unexplored leaf's child is now subjected by the following methods
            let child_node_id = expansion(&mut tree, unexplored_leaf_node.0, unexplored_leaf_node.1);
            let child_node = tree.get(&child_node_id).unwrap().data();

            let child_nodes_nash =
                simulation(*conf, child_node.current_turn, child_node.move_count, &mut simulation_buffer);

            back_propagation(&mut tree, &child_node_id, child_nodes_nash);
        }
    }

    let moves = compute_moves(conf, Phase::from(move_count), current_turn);

    let root_visit_count = tree.get(tree.root_node_id().unwrap()).unwrap().data().move_count;

    // For mapping the configurations on the first tree niveau back to the Move struct needed by the main function
    let selected = tree
        .get(tree.root_node_id().unwrap())
        .unwrap()
        .children()
        .iter()
        .map(|node_id| tree.get(node_id).unwrap().data())
        .map(|mcts_content| mcts_content.calculate_uct(root_visit_count as f64))
        .enumerate()
        .max_by(|(_i1, uct1), (_i2, uct2)| uct1.partial_cmp(uct2).unwrap());

    moves[selected.unwrap().0]
}

fn apply_move(conf: &Configuration, m: &Move, current_turns_color: State) -> Configuration {
    let mut modified_conf = apply_action(conf, m.action, current_turns_color);
    if let Some(to_take) = m.take {
        modified_conf.set(to_take, State::Empty);
    }
    modified_conf
}

/// Selects ChildNode that hasn't been visited, which equals the leftmost child node where calculate_uct is `f64::INFINITY`
fn selection(tree: &Tree<MCTSNodeContent>) -> (NodeId, MCTSNodeContent) {
    let mut current_node_id = tree.root_node_id().unwrap();
    let mut current_node: &MCTSNodeContent = tree.get(current_node_id).unwrap().data();

    // Search the tree along the path with maximal UCT value until node is reached, which wasn't visited so far
    while current_node.visit_count != 0 {
        // Later needed for the calculation of the UCT value of the current nodes children
        let current_nodes_visit_count = current_node.visit_count as f64;

        let currents_child_ids = tree.children_ids(current_node_id).unwrap();

        // Selects a node with the highest UTC-value from the current nodes child nodes
        (current_node_id, current_node) = currents_child_ids
            .map(|node_id| (node_id, tree.get(node_id).unwrap().data()))
            .filter(|(_, node)| !is_config_won_or_lost(&node.conf, Phase::from(node.move_count)))
            .max_by(|(_, child_node1), (_, child_node2)| {
                let uct1 = child_node1.calculate_uct(current_nodes_visit_count);
                let uct2 = child_node2.calculate_uct(current_nodes_visit_count);

                uct1.total_cmp(&uct2)
            })
            .unwrap();
    }

    (current_node_id.clone(), *current_node)
}

/// Adds (all 'suitable') child nodes of the given node n to the tree by applying all possible moves on n
/// Then returns one of the child nodes' NodeId randomly
fn expansion(tree: &mut Tree<MCTSNodeContent>, node_id: NodeId, node: MCTSNodeContent) -> NodeId {
    use id_tree::InsertBehavior::*;

    let child_nodes_turn_color = node.current_turn.flip_color();
    let moves = compute_moves(&node.conf, Phase::from(node.move_count), node.current_turn);

    let child_node_ids: Vec<NodeId> = moves
        .iter()
        .map(|m| apply_move(&node.conf, m, node.current_turn))
        .map(|mod_conf| {
            tree.insert(
                Node::new(MCTSNodeContent::new(mod_conf, node.move_count + 1, child_nodes_turn_color)),
                UnderNode(&node_id),
            )
            .unwrap()
        })
        .collect();

    // TODO use some evaluation function right here to narrow the subtree width
    let i = unsafe { RNG.generate_range(0..child_node_ids.len()) };
    child_node_ids[i].clone()
}

/// Simulates a play-through from the leaf state parameters until one side has won the game & returns the nash value relative
/// to the leaf node the computation started with.
///
/// Uses the simulation_buffer to save some allocations.
fn simulation(
    leaf_conf: Configuration,
    leaf_color: State,
    leaf_move_count: u32,
    simulation_buffer: &mut Vec<Move>,
) -> Nash {
    let mut current_conf = leaf_conf;
    let mut current_color = leaf_color;
    let mut current_move_count = leaf_move_count;

    let mut nash = Nash::DRAW;
    // Extreme case: The unvisited nodes leaf node already is won or lost.
    update_nash_relative_to_leaf(&mut nash, &leaf_conf, leaf_color.flip_color(), leaf_color, Phase::from(leaf_move_count));

    // TODO break the function when the path is too long and there is a high possibility it is a draw
    while nash == Nash::DRAW {
        apply_rand_move(
            &mut current_conf,
            if current_move_count < 18 { Phase::Placement } else { Phase::Moving },
            current_color,
            simulation_buffer,
        );
        simulation_buffer.clear();

        current_move_count += 1;
        current_color = current_color.flip_color();

        update_nash_relative_to_leaf(&mut nash, &current_conf, current_color, leaf_color, Phase::from(leaf_move_count));
    }
    nash
}

/// Computes possible moves from config, chooses one randomly, updates config with new one
// TODO: Evalutation-Function instead of RNG
fn apply_rand_move(conf: &mut Configuration, phase: Phase, color: State, buff: &mut Vec<Move>) {
    for_each_move(conf, phase, color, |m| buff.push(m));

    if buff.is_empty() {
        eprintln!("{conf:?}",);
        panic!()
    }

    let i = unsafe { RNG.generate_range(0..buff.len()) };
    *conf = apply_move(conf, &buff[i], color);
}

/// Calculates nash value for only the current color relative to the leaf color & applies it on the `nash` value argument
/// E.g. White made a move. Now the current_color should be chosen as black to check if black has lost.
///      If so, the nash value is set to either won or lost, respectively to the leaf color which is the node the simulation started with
fn update_nash_relative_to_leaf(
    nash: &mut Nash,
    conf: &Configuration,
    current_color: State,
    leaf_color: State,
    phase: Phase,
) {
    //nash is DRAW by default & stays like that during Placement-Phase due to the game being not decidable yet
    if phase == Phase::Placement {
        return;
    }

    if count(conf, current_color) == 2 || compute_moves(conf, Phase::Moving, current_color).is_empty() {
        if current_color == leaf_color {
            *nash = Nash::LOST;
        } else {
            *nash = Nash::WON;
        }
    }
}

// TODO replace this with the nash calculation in the selection method
fn is_config_won_or_lost(conf: &Configuration, phase: Phase) -> bool {
    phase == Phase::Moving
        && (count(conf, State::Black) == 2
            || count(conf, State::White) == 2
            || compute_moves(conf, Phase::Moving, State::Black).is_empty()
            || compute_moves(conf, Phase::Moving, State::White).is_empty())
}

/* fn back_propagation(tree: &mut Tree<MCTSNodeContent>, leaf_id: NodeId, nash: i32, leaf_turn: State) {
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
} */

// so besser mit mut? weil danach brauchen wir den wert von der leaf nicht oder?
fn back_propagation(tree: &mut Tree<MCTSNodeContent>, start_node_id: &NodeId, nash: Nash) {
    let mut current_node_id = start_node_id.clone();
    let mut current_node = tree.get_mut(&start_node_id).unwrap();

    let mut current_nash = nash;

    loop {
        current_node.data_mut().visit_count += 1;
        if current_nash == Nash::WON {
            current_node.data_mut().win_count += 1;
        }

        (current_node_id, current_node) = if let Ok(ancenstor_ids) = tree.ancestor_ids(&current_node_id) {
            // Any node should have exactly one parent
            let parent_id = ancenstor_ids.last().unwrap();

            (parent_id.clone(), tree.get_mut(&parent_id.clone()).unwrap())
        } else {
            break;
        };

        current_nash = !current_nash;
    }
}
