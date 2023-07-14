use std::{mem::take, ops::Not, time::Instant};

use id_tree::{Node, NodeId, Tree, TreeBuilder};
use nanorand::{Rng, WyRand};
use once_cell::sync::Lazy;

use crate::{
    compute_moves, count,
    ds::{apply_action, Action, Configuration, Move, Phase, State},
    for_each_move,
};

pub static mut RNG: Lazy<WyRand> = Lazy::new(|| WyRand::new());

const TREE_INIT_NODE_COUNT: usize = 2_000_000;

#[derive(PartialEq, Debug)]
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
        if value < 18 {
            Phase::Placement
        } else {
            Phase::Moving
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
struct MCTSNodeContent {
    conf: Configuration,
    move_count: u32,     // to determine the current game phase at every node of the tree
    current_turn: State, // = player_color
    visit_count: u32,    // v_i
    win_count: u32,      // w_i
    is_terminal_node: bool
}

impl MCTSNodeContent {
    /// Initializes the win_count with max because of UCT selection perfering those with win_count == u32::MAX
    fn new(conf: Configuration, move_count: u32, current_turn: State) -> MCTSNodeContent {
        MCTSNodeContent {
            conf,
            move_count,
            current_turn,
            visit_count: 0,
            win_count: u32::MAX,
            is_terminal_node : false,
        }
    }

    /// Computes the value which is used to choose the best path to walk down the tree in the selection-phase
    ///
    /// Takes care of the extreme cases:
    /// - The node was not visited before => INFINITY is returned to use this node next in the select-phase
    /// - The node was visited before, but the win_count == 0 => 0f64 is returned to avoid NaN
    ///   due to 0/v_i of the UTC formula
    fn calculate_uct(&self, parent_node_visit_count: f64) -> f64 {
        return if self.win_count == u32::MAX {
            f64::INFINITY
        }
        // To avoid (0f64 / 1f64) which evaluates to NaN?
        else if self.win_count == 0 {
            0f64
        }
        // Do the actual formula
        else {
            (self.win_count as f64 / self.visit_count as f64)
                + f64::sqrt(f64::ln(parent_node_visit_count) / self.visit_count as f64)
        };
    }
}

/// Calculates Best Move for current Config
pub fn mcts(start_conf: &Configuration, start_move_count: u32, start_turn: State) -> Move {
    let instant = Instant::now();

    let start_node_content = MCTSNodeContent::new(*start_conf, start_move_count, start_turn);

    let mut tree = TreeBuilder::<MCTSNodeContent>::new()
        .with_node_capacity(TREE_INIT_NODE_COUNT)
        .with_root(Node::new(start_node_content))
        .build();

    let start_node_id = tree.root_node_id().unwrap().clone();
    // The start_turn might be black or white, so to determine if the current turn is the same as the start_turn by using
    // a primitive move counter it's rest-class-ring over 2 is needed later
    let start_turn_as_binary = match start_turn {
        State::Empty => u32::MAX,
        State::White => 0,
        State::Black => 1,
    };
    assert!(start_turn_as_binary != u32::MAX);

    let mut simulation_buffer = Vec::<Move>::with_capacity(256);

    while instant.elapsed().as_millis() < 1_000 {
        for _i in 0..10_000 {
            // Perhaps the tree would never be deconstructed, then the niveau we currently are on equals the amount of
            // moves made on the playfield configuration
            // Its not clear, which player color started to move first, but this isn't important
            let mut current_niveau: u32 = start_move_count;

            // Should always pick the most left, non-visited leaf node on the path with the highest UCT value
            let unexplored_leaf_node = selection(&tree, &start_node_id, &mut current_niveau);

            // Extreme case: the selected, unexplored leaf node is some terminal (=already won/ lost) node in which some color already has won
            //               In this case mark it as terminal node to avoid further selection & start the back_propagation on it


            let unexploreds_turn = if current_niveau % 2 == start_turn_as_binary {start_turn} else {start_turn.flip_color()};
            let unexploreds_phase = Phase::from(current_niveau);
            let unexploreds_nash = get_relative_nash(&unexplored_leaf_node.1.conf, unexploreds_turn, start_turn, unexploreds_phase);

            if unexploreds_nash != Nash::DRAW {
                tree.get_mut(&unexplored_leaf_node.0).unwrap().data_mut().is_terminal_node = true;

                back_propagation(&mut tree, &unexplored_leaf_node.0, unexploreds_nash);
                continue;
            }

            // A child node of the unexplored leaf node is now subjected by the following methods
            let child_node_id = expansion(&mut tree, unexplored_leaf_node.0, unexplored_leaf_node.1);
            let child_node = tree.get(&child_node_id).unwrap().data();

            assert!(child_node.visit_count == 0);
            current_niveau += 1;

            let child_nodes_turn = unexploreds_turn.flip_color();
            let child_nodes_phase = Phase::from(current_niveau);
            let child_nodes_nash = get_relative_nash(&child_node.conf, child_nodes_turn, start_turn, child_nodes_phase);

            if child_nodes_nash != Nash::DRAW {
                tree.get_mut(&child_node_id).unwrap().data_mut().is_terminal_node = true;

                back_propagation(&mut tree, &child_node_id, child_nodes_nash);
                continue;
            }

            let child_nodes_nash =
                simulation(*start_conf, child_nodes_turn, current_niveau, &mut simulation_buffer);

            back_propagation(&mut tree, &child_node_id, child_nodes_nash);
        }
    }

    let root_visit_count = tree.get(tree.root_node_id().unwrap()).unwrap().data().move_count as f64;

    // For mapping the configurations on the first tree niveau back to the Move struct needed by the main function
    let selected = tree
        .get(tree.root_node_id().unwrap())
        .unwrap()
        .children()
        .iter()
        .map(|node_id| tree.get(node_id).unwrap().data())
        .max_by(|node1, node2| {
            let uct1 = node1.calculate_uct(root_visit_count);
            let uct2 = node2.calculate_uct(root_visit_count);
            uct1.partial_cmp(&uct2).unwrap()
        })
        .unwrap();

    eprintln!(
        "Selected playfield with UCT: {}\nroot visit count: {}\nmcts node values: \n\tw_i: {}\n\tv_i: {}",
        selected.calculate_uct(root_visit_count),
        root_visit_count,
        selected.win_count,
        selected.visit_count
    );

    // TODO map the config from selected back to move? Maybe better solution I cant think of atm.
    // TODO Commented out lead to out of bounds, strangely. Is the order of the nodes inserted into the id-tree random or are there maybe more nodes added to the id tree than the ones from `calculate_moves`???

    convert_config_to_move(*start_conf, start_turn, selected.conf)
}

/// Applies a random move out of the possible ones
fn apply_move(conf: &Configuration, m: &Move, current_turns_color: State) -> Configuration {
    let mut modified_conf = apply_action(conf, m.action, current_turns_color);
    if let Some(to_take) = m.take {
        modified_conf.set(to_take, State::Empty);
    }
    modified_conf
}

/// Selects most promissing leaf node in the tree that hasn't been visited & is not a terminal position,
/// which equals the leftmost child node of the path in the tree with the highest UCT value.
///
/// Uses the `current_niveau` variable to keep track on which niveau the returned node tuple is on, which is important for
/// determination of the game phase & nash value
fn selection(tree: &Tree<MCTSNodeContent>, start_node: &NodeId, current_niveau: &mut u32) -> (NodeId, MCTSNodeContent) {
    let mut current_node_id: &NodeId = start_node;
    let mut current_node: &MCTSNodeContent = tree.get(start_node).unwrap().data();

    // Search the tree along the path with maximal UCT value until node is reached, which wasn't visited so far
    // as long as current_node.visit_count != 0 || 1 due to
    while !(0..=1).contains(&current_node.visit_count) {
        *current_niveau += 1;

        // Later needed for the calculation of the UCT value of the current nodes children
        let current_nodes_visit_count = current_node.visit_count as f64;

        let currents_child_ids = tree.children_ids(current_node_id).unwrap();

        // Selects a node with the highest UTC-value from the current nodes child nodes
        // Terminal configs should generally be filtered because of UTC... ?! TODO
        let child_node_with_max_utc = currents_child_ids
            .map(|node_id| (node_id, tree.get(node_id).unwrap().data()))
            .filter(|(_, node)| !node.is_terminal_node)
            //.filter(|(_, node)| !is_config_won_or_lost(&node.conf, Phase::from(node.move_count)))
            .max_by(|(_, child_node1), (_, child_node2)| {
                let uct1 = child_node1.calculate_uct(current_nodes_visit_count);
                let uct2 = child_node2.calculate_uct(current_nodes_visit_count);

                uct1.total_cmp(&uct2)
            })
            .unwrap();

        (current_node_id, current_node) = child_node_with_max_utc;
    }

    (current_node_id.clone(), *current_node)
}

/// Adds (all 'suitable') child nodes of the given node n to the tree by applying all possible moves on n
/// Then returns one of the child nodes' NodeId randomly
fn expansion(tree: &mut Tree<MCTSNodeContent>, node_id: NodeId, node: MCTSNodeContent) -> NodeId {
    let child_nodes_turn_color = node.current_turn.flip_color();
    let moves = compute_moves(&node.conf, Phase::from(node.move_count), node.current_turn);

    // Inserting the possible moves applied on the current nodes Config into the tree as the current nodes children
    let child_node_ids: Vec<NodeId> = moves
        .iter()
        .map(|m| apply_move(&node.conf, m, node.current_turn))
        .map(|mod_conf| {
            tree.insert(
                Node::new(MCTSNodeContent::new(mod_conf, node.move_count + 1, child_nodes_turn_color)),
                id_tree::InsertBehavior::UnderNode(&node_id),
            )
            .unwrap()
        })
        .collect();

    // TODO use some evaluation function right here to narrow the subtree width
    let i = unsafe { RNG.generate_range(0..child_node_ids.len()) };
    child_node_ids[i].clone()
}

/// Simulates a play-through from the expansion nodes state parameters until one side has won the game &
/// returns the nash value relative to the leaf node the computation started with.
///
/// Uses the simulation_buffer to save some allocations.
fn simulation(
    expanded_nodes_conf: Configuration,
    expanded_nodes_turn: State,
    expanded_nodes_move_count: u32,
    simulation_buffer: &mut Vec<Move>,
) -> Nash {
    let mut current_conf = expanded_nodes_conf;
    let mut current_color = expanded_nodes_turn;
    let mut current_move_count = expanded_nodes_move_count;

    // Extreme case: The extended node already is won or lost
    let mut current_nash = get_relative_nash(
        &expanded_nodes_conf,
        expanded_nodes_turn.flip_color(),
        expanded_nodes_turn,
        Phase::from(expanded_nodes_move_count),
    );

    // TODO break the function when the path is too long and there is a high possibility it is a draw
    // let mut a = 0;
    while current_nash == Nash::DRAW {
        /* eprintln!("{a}");
        a += 1; */

        apply_rand_move(&mut current_conf, Phase::from(expanded_nodes_move_count), current_color, simulation_buffer);
        simulation_buffer.clear();

        current_move_count += 1;
        current_color = current_color.flip_color();

        current_nash = get_relative_nash(
            &current_conf,
            current_color,
            expanded_nodes_turn,
            Phase::from(current_move_count),
        );
    }
    current_nash
}

/// Computes possible moves from config, chooses one randomly, updates config with new one
// TODO: Evalutation-Function instead of RNG
fn apply_rand_move(conf: &mut Configuration, phase: Phase, color: State, buff: &mut Vec<Move>) {
    for_each_move(conf, phase, color, |m| buff.push(m));

    if buff.is_empty() {
        eprintln!("{conf:?}",);
    }

    let i = unsafe { RNG.generate_range(0..buff.len()) };
    *conf = apply_move(conf, &buff[i], color);
}

/// Calculates the nash value for only the current turn relative to the other turn. It checks if the conf is won or
/// lost by probing if the current_turns color would be able to make it's next move.
///
/// E.g. White made a move. Now the current_color should be chosen as black to check if black has lost.
///      If so, the return value is set to either won or lost, respectively to the other color which is the node the simulation started with
fn get_relative_nash(
    conf: &Configuration,
    current_turn: State,
    other_turn: State,
    phase_curent: Phase,
) -> Nash {
    // placement-phase is never a terminal state
    if phase_curent == Phase::Placement {
        return Nash::DRAW;
    }

    return if count(conf, current_turn) == 2 || !are_moves_possible(conf, Phase::Moving, current_turn) {
        if current_turn == other_turn {
            Nash::LOST
        } else {
            Nash::WON
        }
    } else {
        Nash::DRAW
    }
}

fn are_moves_possible(conf: &Configuration, phase: Phase, current_color: State) -> bool {
    let mut moves_possible = false;
    for_each_move(conf, phase, current_color, |_| moves_possible = true);
    moves_possible
}

// TODO replace this with the nash calculation in the selection method
/// Returns color who won
fn is_config_won_or_lost(conf: &Configuration, phase: Phase) -> Option<State> {

    if phase == Phase::Placement {return Option::None}

    if count(conf, State::Black) == 2 {return Option::Some(State::White)}
    else if count(conf, State::White) == 2 {return Option::Some(State::Black)}
    else if compute_moves(conf, Phase::Moving, State::Black).is_empty() {return Option::Some(State::White)}
    else if compute_moves(conf, Phase::Moving, State::White).is_empty() {return Option::Some(State::Black)}
    else {return Option::None}
}

// so besser mit mut? weil danach brauchen wir den wert von der leaf nicht oder?
fn back_propagation(tree: &mut Tree<MCTSNodeContent>, leaf_node_id: &NodeId, nash: Nash) {
    let mut current_node_id = leaf_node_id.clone();
    let mut current_node = tree.get_mut(&leaf_node_id).unwrap().data_mut();

    let mut current_nash = nash;

    loop {
        current_node.visit_count += 1;

        // Due to new nodes being initialized with u32::MAX, we have to manually reset them to 1/ 0 when visiting
        // them the first time
        if current_nash == Nash::WON {
            if current_node.win_count == u32::MAX
            /* == was_unvisited_node */
            {
                current_node.win_count = 1;
            } else {
                current_node.win_count += 1;
            }
        } else {
            if current_node.win_count == u32::MAX
            /* == was_unvisited_node */
            {
                current_node.win_count = 0;
            }
        }

        match tree.ancestor_ids(&current_node_id) {
            Ok(mut ancenstor_ids) => {
                if let Some(parent_id) = ancenstor_ids.next() {
                    current_node_id = parent_id.clone();
                    current_node = tree.get_mut(&parent_id.clone()).unwrap().data_mut();
                } else {
                    break;
                }
            }
            Err(_) => break,
        }
        current_nash = !current_nash;
    }
}

/// Due to Configuration being saved in the tree and a Move is needed by the protocol
fn convert_config_to_move(config_first: Configuration, turn_color: State, config_second: Configuration) -> Move {
    let mismatches: Vec<(usize, (&State, &State))> = config_first
        .arr
        .iter()
        .flatten()
        .zip(config_second.arr.iter().flatten())
        .enumerate()
        .filter(|(_, (state_first, state_second))| state_first != state_second)
        .collect();

    assert!(1 <= mismatches.len() && mismatches.len() < 4);

    // The field index something was placed upon on the first config, but the second one has nothing on it
    let moved_stone: Vec<usize> =
        mismatches.iter().filter(|(_, (state1, _))| **state1 == turn_color).map(|tuple| tuple.0).collect();
    assert!(moved_stone.len() < 2);

    // Previously nothing, now something
    let placed_stone: Vec<usize> =
        mismatches.iter().filter(|(_, (_, state2))| **state2 == turn_color).map(|tuple| tuple.0).collect();
    assert!(placed_stone.len() == 1);

    // Previously something, now nothing but different colored stone
    let taken_stone: Vec<usize> = mismatches
        .iter()
        .filter(|(_, (state1, state2))| **state1 == turn_color.flip_color() && **state2 == State::Empty)
        .map(|tuple| tuple.0)
        .collect();
    assert!(taken_stone.len() < 2);

    let place_field_index = ((placed_stone[0] / 8) as u8, (placed_stone[0] % 8) as u8);

    let action = if moved_stone.len() == 1 {
        Action::Move(((moved_stone[0] / 8) as u8, (moved_stone[0] % 8) as u8), place_field_index)
    } else {
        Action::Place(place_field_index)
    };

    return Move {
        action,
        take: if taken_stone.len() == 1 {
            Some((((taken_stone[0] / 8) as u8), ((taken_stone[0] % 8) as u8)))
        } else {
            Option::None
        },
    };
}
