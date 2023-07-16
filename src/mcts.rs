use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use id_tree::{InsertBehavior, Node, NodeId, Tree};
use nanorand::{Rng, WyRand};
use once_cell::sync::Lazy;

use crate::{
    count,
    ds::{apply_action, Action, Configuration, Move, Phase, State},
    for_each_move, is_corner, is_muhle,
};

pub static mut RNG: Lazy<WyRand> = Lazy::new(|| WyRand::new());

pub(crate) const TREE_INIT_NODE_COUNT: usize = 2_000_000;
const IN_THREAD_SIMULATIONS: usize = 2;
const IN_THREAD_BUFF_SIZE: usize = 256;

#[derive(PartialEq, Debug)]
enum Nash {
    WON,
    DRAW,
    LOST,
}

impl std::ops::Not for Nash {
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
pub(crate) struct MCTSNodeContent {
    pub(crate) conf: Configuration,
    visit_count: u32, // v_i
    win_count: u32,   // w_i
    is_terminal_node: bool,
}

impl MCTSNodeContent {
    /// Initializes the win_count with max because of UCT selection perfering those with win_count == u32::MAX
    pub fn new(conf: Configuration, win_count: u32, visit_count: u32) -> MCTSNodeContent {
        MCTSNodeContent { conf, visit_count, win_count, is_terminal_node: false }
    }

    /// Computes the value which is used to choose the best path to walk down the tree in the selection-phase
    ///
    /// Takes care of the extreme cases:
    /// - The node was not visited before => INFINITY is returned to use this node next in the select-phase
    /// - The node was visited before, but the win_count == 0 => 0f64 is returned to avoid NaN
    ///   due to 0/v_i of the UTC formula
    fn calculate_uct(&self, parent_node_visit_count: f64) -> f64 {
        if self.win_count == u32::MAX {
            f64::INFINITY
        }
        // To avoid (0f64 / 1f64) which evaluates to NaN?
        else if self.win_count == 0 {
            if self.visit_count != 0 && parent_node_visit_count != 0f64 {
                f64::sqrt(f64::ln(parent_node_visit_count) / self.visit_count as f64)
            } else {
                0f64
            }
        }
        // Do the actual formula
        else {
            (self.win_count as f64 / self.visit_count as f64) + f64::sqrt(f64::ln(parent_node_visit_count) / self.visit_count as f64)
        }
    }
}

/// Calculates Best Move for current Config
pub(crate) fn mcts(
    mut tree: Tree<MCTSNodeContent>,
    last_mcts_pick_node_id: &mut Option<NodeId>,
    start_conf: &Configuration,
    start_move_count: u32,
    start_turn: State,
) -> (Move, NodeId, Tree<MCTSNodeContent>) {
    let simulation_threads: usize = <NonZeroUsize as Into<usize>>::into(thread::available_parallelism().unwrap());
    let instant = Instant::now();

    let start_node_id = match last_mcts_pick_node_id {
        // Normal case, when the root of the tree is already present & the play is somewhere in already set in move
        Some(last_mcts_pick_node_id) => {
            // Looks if the move the opponent made is already in the tree
            let start_node_in_tree_maybe = tree.children_ids(last_mcts_pick_node_id)
                .unwrap()
                .find(|child_node_id| tree.get(child_node_id).unwrap().data().conf == *start_conf);

            // If the opponent move made is not in the tree, insert a new node with the start_conf into the tree
            match start_node_in_tree_maybe {
                Some(node) => node.clone(),
                None => tree
                    .insert(
                        Node::new(MCTSNodeContent::new(*start_conf, u32::MAX, 0)),
                        InsertBehavior::UnderNode(&last_mcts_pick_node_id),
                    )
                    .unwrap(),
            }
        }
        // Extreme case: The tree is empty. No root node is set, because it's not clear if we opening the game...
        None => tree
            .insert(
                Node::new(MCTSNodeContent {
                    conf: *start_conf,
                    visit_count: 0,
                    win_count: u32::MAX,
                    is_terminal_node: false,
                }),
                InsertBehavior::AsRoot,
            )
            .unwrap(),
    };

    // The start_turn might be black or white, so to determine if the current turn is the same as the start_turn by using
    // a primitive move counter it's rest-ring-class over 2 is needed later
    let start_turn_as_binary = match start_turn {
        State::Empty => panic!(),
        State::White => 0,
        State::Black => 1,
    };

    let mut conf_buf = Vec::<(Configuration, u32)>::with_capacity(256);

    'outer: while instant.elapsed().as_millis() < 1_000 {
        for _i in 0..1_000 {
            // Perhaps the tree would never be deconstructed, then the niveau we currently are on equals the amount of
            // moves made on the playfield configuration
            // Its not clear, which player color started to move first, but this isn't important
            let mut current_niveau: u32 = start_move_count;

            // Should always pick the most left, non-visited leaf node on the path with the highest UCT value
            let (unexplored_leaf_node_id, unexplored_leaf_node) =
                match selection(&mut tree, &start_node_id, &mut current_niveau) {
                    Some(tuple) => tuple,
                    None => break 'outer,
                };

            let unexp_turn =
                if current_niveau % 2 == start_turn_as_binary { start_turn } else { start_turn.flip_color() };
            let unexp_phase = Phase::from(current_niveau);
            let unexp_nash = get_relative_nash(&unexplored_leaf_node.conf, unexp_turn, start_turn, unexp_phase);

            // Extreme case: the selected, unexplored leaf node is some terminal node (=already won/ lost).
            // In this case mark it as terminal node to avoid further selection & start the back_propagation on it
            if unexp_nash != Nash::DRAW {
                tree.get_mut(&unexplored_leaf_node_id).unwrap().data_mut().is_terminal_node = true;

                back_propagation(&mut tree, &start_node_id, &unexplored_leaf_node_id, unexp_nash);
                continue;
            }

            // A child node of the unexplored leaf node is now subjected by the following methods
            let child_node_id = expansion(
                &mut tree,
                unexplored_leaf_node_id,
                unexplored_leaf_node,
                current_niveau,
                unexp_turn,
                &mut conf_buf,
            );
            conf_buf.clear();

            let child_node = tree.get(&child_node_id).unwrap().data().clone();
            assert!(child_node.visit_count == 0 || child_node.visit_count == 1);

            current_niveau += 1;

            if Phase::from(start_move_count + 2) == Phase::Placement {
                return (convert_config_to_move(start_conf.clone(), start_turn, child_node.conf), child_node_id, tree);
            }

            // Extreme case from above, now for a niveau + 1...
            let child_nodes_turn = unexp_turn.flip_color();
            let child_nodes_phase = Phase::from(current_niveau);
            let mut child_nodes_nash =
                get_relative_nash(&child_node.conf, child_nodes_turn, start_turn, child_nodes_phase);

            if child_nodes_nash != Nash::DRAW {
                tree.get_mut(&child_node_id).unwrap().data_mut().is_terminal_node = true;
            } else {
                // Running multiple simulations at once with the available amount of thread power!
                let nash_median = Arc::new(Mutex::new(0f64));
                let mut threads_results = Vec::with_capacity(simulation_threads);

                for _ in 0..simulation_threads {
                    let nash = nash_median.clone();
                    threads_results.push(thread::spawn(move || {
                        let mut local_rng = WyRand::new();

                        let mut buff = Vec::<(Configuration, u32)>::with_capacity(IN_THREAD_BUFF_SIZE);
                        let mut local_nash_median = 0f64;

                        for _ in 0..IN_THREAD_SIMULATIONS {
                            let child_nodes_nash = simulation(
                                child_node.conf,
                                child_nodes_turn,
                                current_niveau,
                                &mut local_rng,
                                &mut buff,
                            );
                            local_nash_median += if child_nodes_nash == Nash::WON { 1f64 } else { -1f64 };
                        }

                        let mut lock = nash.lock().unwrap();
                        *lock += local_nash_median;
                    }))
                }

                // Joining the threads back together
                for thread_result in threads_results {
                    thread_result.join().unwrap()
                }

                child_nodes_nash =
                    if *nash_median.lock().unwrap() > (IN_THREAD_SIMULATIONS * simulation_threads) as f64 * -0.1f64
                    {
                        Nash::WON
                    } else {
                        Nash::LOST
                    };
            }

            back_propagation(&mut tree, &start_node_id, &child_node_id, child_nodes_nash);
        }
    }

    let start_visit_count = tree.get(&start_node_id).unwrap().data().visit_count as f64;

    // For mapping the configurations on the first tree niveau back to the Move struct needed by the main function
    let (selected_node, selected_node_id) = tree
        .get(&start_node_id)
        .unwrap()
        .children()
        .iter()
        .map(|node_id| (tree.get(node_id).unwrap().data(), node_id))
        .max_by(|(node1, _node_id1), (node2, _node_id2)| {
            let uct1 = node1.calculate_uct(start_visit_count);
            let uct2 = node2.calculate_uct(start_visit_count);

            uct1.partial_cmp(&uct2).unwrap()
        })
        .unwrap();

    eprintln!("\nAll available child nodes on of the start_node with their weightings:");
    tree
        .get(&start_node_id)
        .unwrap()
        .children()
        .iter()
        .map(|node_id| tree.get(node_id).unwrap().data())
        .for_each(|n| eprintln!("v_i: {}, w_i: {}, utc: {}, terminal: {}", n.visit_count, n.win_count, n.calculate_uct(start_visit_count), n.is_terminal_node));

    eprintln!(
        "---\nSelected playfield with UCT: {}\nstart node visit count: {}\nmcts node values: \n\tw_i: {}\n\tv_i: {}",
        selected_node.calculate_uct(start_visit_count),
        start_visit_count,
        selected_node.win_count,
        selected_node.visit_count
    );

    let picked_move = convert_config_to_move(*start_conf, start_turn, selected_node.conf);
    (picked_move, selected_node_id.clone(), tree)
}

/// Applies a random move out of the possible ones
pub(crate) fn apply_move(conf: &Configuration, m: &Move, current_turns_color: State) -> Configuration {
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
fn selection(
    tree: &mut Tree<MCTSNodeContent>,
    start_node: &NodeId,
    current_niveau: &mut u32,
) -> Option<(NodeId, MCTSNodeContent)> {
    let mut current_node_id: NodeId = start_node.clone();
    let mut current_node: MCTSNodeContent = tree.get(start_node).unwrap().data().clone();

    // Search the tree along the path with maximal UCT value until node is reached, which wasn't visited so far
    // as long as current_node.visit_count != 0 || 1 due to
    while current_node.visit_count != 0 {
        *current_niveau += 1;

        // Later needed for the calculation of the UCT value of the current nodes children
        let current_nodes_visit_count = current_node.visit_count as f64;

        let currents_child_ids = tree.children_ids(&current_node_id).unwrap();

        // Selects a node with the highest UTC-value from the current nodes child nodes
        // Terminal configs should generally be filtered because of UTC
        let child_node_with_max_utc = currents_child_ids
            .map(|node_id| (node_id, tree.get(node_id).unwrap().data()))
            .filter(|(_, node)| !node.is_terminal_node)
            .max_by(|(_, child_node1), (_, child_node2)| {
                let uct1 = child_node1.calculate_uct(current_nodes_visit_count);
                let uct2 = child_node2.calculate_uct(current_nodes_visit_count);

                match uct1.partial_cmp(&uct2) {
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Greater,
                }
            });

        // Extreme case: The parent current node only contains childs which are terminal nodes. Therefore the current
        // node must also be flagged as is_terminal_node to avoid being on the selection path the next iteration
        (current_node_id, current_node) = match child_node_with_max_utc {
            Some(node_tuple) => (node_tuple.0.clone(), node_tuple.1.clone()),
            None => {
                tree.get_mut(&current_node_id).unwrap().data_mut().is_terminal_node = true;

                let parent_node_id = match tree.ancestor_ids(&current_node_id).unwrap().next() {
                    Some(parent) => parent.clone(),
                    None => return Option::None,
                };
                let parent_node = tree.get(&parent_node_id).unwrap().data().clone();

                (parent_node_id, parent_node)
            }
        };
    }

    Option::Some((current_node_id.clone(), current_node))
}

/// Adds (all 'suitable') child nodes of the given node n to the tree by applying all possible moves on n
/// Then returns one of the child nodes' NodeId randomly
fn expansion(
    tree: &mut Tree<MCTSNodeContent>,
    node_id: NodeId,
    mut node: MCTSNodeContent,
    move_count: u32,
    current_turn: State,
    buff: &mut Vec<(Configuration, u32)>,
) -> NodeId {
    compute_best_configs(&mut node.conf, Phase::from(move_count), current_turn, buff);

    // In the placement-phase, we don"t want to call simulation or pick one randomly
    // We want the best move according to the simulation function and just return it
    if Phase::from(move_count) == Phase::Placement {
        let node_with_best_rating = buff
            .iter()
            .map(|(conf, rating)| {
                let node_id = tree
                    .insert(
                        Node::new(MCTSNodeContent::new(*conf, *rating, 0)),
                        id_tree::InsertBehavior::UnderNode(&node_id),
                    )
                    .unwrap();

                (rating, node_id)
            })
            .max_by(|(rating1, _), (rating2, _)| rating1.cmp(rating2))
            .unwrap();

        return node_with_best_rating.1.clone();
    } else {
        // Inserting the possible moves applied on the current nodes Config into the tree as the current nodes children
        let child_node_ids: Vec<NodeId> = buff
            .iter()
            .map(|(conf, rating)| {
                tree.insert(
                    Node::new(MCTSNodeContent::new(*conf, *rating, 0)),
                    id_tree::InsertBehavior::UnderNode(&node_id),
                )
                .unwrap()
            })
            .collect();

        let i = unsafe { RNG.generate_range(0..buff.len()) };
        return child_node_ids[i].clone();
    };
}

fn compute_best_configs(
    conf: &mut Configuration,
    phase: Phase,
    current_turn: State,
    buff: &mut Vec<(Configuration, u32)>,
) {
    let mut sum: u32 = 0;

    for_each_move(&conf.clone(), phase, current_turn, |m| {
        let (conf, rating) = evaluate_move(conf, m, phase, current_turn);
        sum += rating;

        buff.push((conf, rating));
    });

    let median = sum / buff.len() as u32; // TODO maybe use percentile (=1/2, 1/3) cut-off instead of median???

    *buff = buff
        .iter()
        .map(|tuple| *tuple)
        .filter(|(_, rating)| median <= *rating)
        //.filter(|(_, rating)| if buff.len() as i32 <= sum { median <= *rating } else { median < *rating })
        .collect();
}

fn evaluate_move(conf: &mut Configuration, m: Move, phase: Phase, current_turn: State) -> (Configuration, u32) {
    let mut rating: u32 = 1;

    let target_position = match m.action {
        Action::Place(place_pos) => place_pos,
        Action::Move(_, target_pos) => target_pos,
    };

    let mod_conf = apply_move(conf, &m, current_turn);

    let oponent_move_simulation = apply_move(conf, &m, current_turn.flip_color());

    // mill check self
    let opponent_stone_count = count(conf, current_turn.flip_color());
    let opponent_stone_count_after_move = count(&mod_conf, current_turn.flip_color());

    let mut move_blocks_oponent_mill = false;
    if opponent_stone_count > opponent_stone_count_after_move {
        rating += 10;
    }
    // Avoid a mill of the opponent by using placing a stone in a two stone mill
    else if is_muhle(&oponent_move_simulation, target_position) {
        move_blocks_oponent_mill = true;
        rating += 2;
        if opponent_stone_count == 3 {
            rating += 10;
        }
    }

    if phase == Phase::Placement {
        if move_blocks_oponent_mill {
            rating += 5;
        }

        if !is_corner(target_position) {
            rating += 1;

            // if there is one empty field & one occupied with some of the same as current_turn, either on or across rings
            let on_ring_state_next = conf.arr[target_position.0 as usize][((target_position.1 + 1) % 8) as usize];
            let on_ring_state_previous = conf.arr[target_position.0 as usize][((target_position.1 + 7) % 8) as usize];

            if (on_ring_state_next == current_turn && on_ring_state_previous == State::Empty)
                | (on_ring_state_next == State::Empty && on_ring_state_previous == current_turn)
            {
                rating += 3;
            }

            let across_rings_state_next = conf.arr[((target_position.0 + 1) % 3) as usize][target_position.1 as usize];
            let across_rings_state_previous =
                conf.arr[((target_position.0 + 2) % 3) as usize][target_position.1 as usize];

            if (across_rings_state_next == current_turn && across_rings_state_previous == State::Empty)
                | (across_rings_state_next == State::Empty && across_rings_state_previous == current_turn)
            {
                rating += 3;
            }
        } else {
            // check corner placement for possible muehle trap setup
            let next_on_ring_state = conf.arr[target_position.0 as usize][((target_position.1 + 1) % 8) as usize];
            let next_next_on_ring_state = conf.arr[target_position.0 as usize][((target_position.1 + 2) % 8) as usize];

            let previous_on_ring_state = conf.arr[target_position.0 as usize][((target_position.1 - 1) % 8) as usize];
            let previous_previous_on_ring_state =
                conf.arr[target_position.0 as usize][((target_position.1 + 6) % 8) as usize];

            if ((next_on_ring_state == current_turn && next_next_on_ring_state == State::Empty)
                | (next_on_ring_state == State::Empty && next_next_on_ring_state == current_turn))
                && ((previous_on_ring_state == current_turn && previous_previous_on_ring_state == State::Empty)
                    | (previous_on_ring_state == State::Empty && previous_previous_on_ring_state == current_turn))
            {
                rating += 2;
            }
        }
    } else {
        //let Vec<Configuration> = compute_moves(&mod_conf, phase, current_turn.flip_color());

        // favor moves that open mills
        if let Action::Move(start_position, _) = m.action {
            if is_muhle(conf, start_position) {
                let mut opponent_can_mill = false;
                let mut opponent_can_block_mill = false;

                for_each_move(&mod_conf, phase, current_turn.flip_color(), |m| {
                    let opponent_target_pos =
                        if let Action::Move(_, target_position) = m.action { target_position } else { panic!() };

                    if is_muhle(&mod_conf, opponent_target_pos) {
                        opponent_can_mill = true;
                    } else if start_position == opponent_target_pos {
                        opponent_can_block_mill = true;
                    }
                });

                if !opponent_can_mill && !opponent_can_block_mill {
                    rating += 10;
                }
            }
        }
    }
    (mod_conf, rating)
}

/// Simulates a play-through from the expansion nodes state parameters until one side has won the game &
/// returns the nash value relative to the leaf node the computation started with.
///
/// Uses the simulation_buffer to save some allocations.
fn simulation(
    expanded_nodes_conf: Configuration,
    expanded_nodes_turn: State,
    expanded_nodes_move_count: u32,
    rng: &mut WyRand,
    conf_buf: &mut Vec<(Configuration, u32)>,
) -> Nash {
    let mut current_conf = expanded_nodes_conf;
    let mut current_turn = expanded_nodes_turn;
    let mut current_move_count = expanded_nodes_move_count;

    let mut current_nash = Nash::DRAW;

    // TODO break the function when the path is too long and there is a high possibility it is a draw
    while current_nash == Nash::DRAW {
        current_conf = get_rand_move(&mut current_conf, Phase::from(current_move_count), current_turn, rng, conf_buf);
        conf_buf.clear();

        current_move_count += 1;
        current_turn = current_turn.flip_color();

        current_nash =
            get_relative_nash(&current_conf, current_turn, expanded_nodes_turn, Phase::from(current_move_count));
    }
    current_nash
}

/// Computes possible moves from config, chooses one randomly, updates config with new one
fn get_rand_move(
    conf: &mut Configuration,
    phase: Phase,
    current_turn: State,
    rng: &mut WyRand,
    buff: &mut Vec<(Configuration, u32)>,
) -> Configuration {
    compute_best_configs(conf, phase, current_turn, buff);

    let i = rng.generate_range(0..buff.len());
    buff[i].0
}

/// Calculates the nash value for only the current turn relative to the other turn. It checks if the conf is won or
/// lost by probing if the current_turns color would be able to make it's next move.
///
/// E.g. White made a move. Now the current_color should be chosen as black to check if black has lost.
///      If so, the return value is set to either won or lost, respectively to the other color which is the node the simulation started with
fn get_relative_nash(conf: &Configuration, current_turn: State, other_turn: State, phase_curent: Phase) -> Nash {
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
    };
}

fn are_moves_possible(conf: &Configuration, phase: Phase, current_color: State) -> bool {
    let mut moves_possible = false;
    for_each_move(conf, phase, current_color, |_| moves_possible = true);
    moves_possible
}

// so besser mit mut? weil danach brauchen wir den wert von der leaf nicht oder?
fn back_propagation(tree: &mut Tree<MCTSNodeContent>, top_node_id: &NodeId, leaf_node_id: &NodeId, nash: Nash) {
    let mut current_node_id = leaf_node_id.clone();
    let mut current_node = tree.get_mut(&leaf_node_id).unwrap().data_mut();

    let mut current_nash = nash;

    loop {
        if current_node_id != *leaf_node_id {
            current_node.visit_count += 1;

            // Due to new nodes being initialized with u32::MAX, we have to manually reset them to 1/ 0 when visiting
            // them the first time
            if current_nash == Nash::WON {
                // == was_unvisited_node
                if current_node.win_count == u32::MAX {
                    current_node.win_count = 1;
                } else {
                    current_node.win_count += 1;
                }
            } else {
                // == was_unvisited_node
                if current_node.win_count == u32::MAX {
                    current_node.win_count = 0;
                }
            }
        }

        match tree.ancestor_ids(&current_node_id).unwrap().next() {
            Some(parent_id) => {
                let parent_is_top_node = parent_id == top_node_id;

                current_node_id = parent_id.clone();
                current_node = tree.get_mut(&parent_id.clone()).unwrap().data_mut();

                // The parent node must too be updated before returing this function
                if parent_is_top_node {
                    current_node.visit_count += 1;

                    current_node.win_count = if current_nash == Nash::LOST {
                        // == was_unvisited_node
                        if current_node.win_count == u32::MAX {
                            1
                        } else {
                           current_node.win_count + 1
                        }
                    } else {
                        // == was_unvisited_node
                        if current_node.win_count == u32::MAX {
                            0
                        } else {
                            current_node.win_count
                        }
                    };
                    return;
                }
            }
            // This formerly was a panic, and this should actually never happen, but im not sure whats going on at this
            // Point so im throwing a return and am happy i guess :)
            None => return,
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
