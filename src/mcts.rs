use id_tree::{Node, NodeId, Tree, TreeBuilder};
use nanorand::{Rng, WyRand};
use once_cell::sync::Lazy;

use crate::{
    compute_moves, count,
    ds::{apply_action, Configuration, Move, Phase, State},
};

pub static mut RNG: Lazy<WyRand> = Lazy::new(|| WyRand::new());

#[derive(PartialEq)]
struct MCTSNode {
    config: Configuration,
    //phase: Phase,
    current_turn: State,
    visitCount: u32, // v_i
    winCount: u32,   // w_i
}

impl MCTSNode {
    fn new(config: Configuration, phase: Phase, current_turn: State) -> MCTSNode {
        MCTSNode {
            config,
            /* phase , */ current_turn,
            visitCount: 0,
            winCount: 0,
        }
    }

    fn calculate_uct(&self, parrent_node_visit_ount: f64) -> f64 {
        (self.winCount as f64 / self.visitCount as f64)
            + f64::sqrt(f64::ln(parrent_node_visit_ount) / self.visitCount as f64)
    }
}

pub fn mcts(config: &Configuration, phase: Phase, possible_moves: &Vec<Move>, roots_current_turn: State) {
    use id_tree::InsertBehavior::*;

    let mut tree = TreeBuilder::<MCTSNode>::new()
        .with_node_capacity(possible_moves.len())
        .with_root(Node::new(MCTSNode::new(*config, phase, roots_current_turn)))
        .build();

    let root_id = tree.root_node_id().unwrap().to_owned();

    for m in possible_moves {
        let modified_config = apply_move(config, m, roots_current_turn);

        let new_node = Node::new(MCTSNode::new(modified_config, phase, roots_current_turn));
        tree.insert(new_node, UnderNode(&root_id)).unwrap();
    }

    // As long as time is available, do:
    loop {
        let node_max_UCT = selction(&tree);
        let nodes_expansion_id = expansion(&mut tree, node_max_UCT);
        let nash = simulation(*config, tree.get(&nodes_expansion_id).unwrap().data().current_turn, roots_current_turn);
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

fn selction(tree: &Tree<MCTSNode>) -> (NodeId, &MCTSNode) {
    let mut current_node_id = tree.root_node_id().unwrap();
    let mut current_node = tree.get(current_node_id).unwrap();

    // Search the tree along the path with maximal UCT value until node is reached, which not was visited so far
    while current_node.data().visitCount != 0 {
        let anc_ids = tree.ancestor_ids(current_node_id).unwrap();

        let next_node = anc_ids
            .max_by(|id1, id2| {
                let parent_visit_count = current_node.data().visitCount as f64;

                let node1_uct = tree.get(id1).unwrap().data().calculate_uct(parent_visit_count);
                let node2_uct = tree.get(id2).unwrap().data().calculate_uct(parent_visit_count);

                node1_uct.total_cmp(&node2_uct)
            })
            .unwrap();

        current_node_id = next_node;
        current_node = tree.get(current_node_id).unwrap();
    }

    (current_node_id.clone(), current_node.data())
}

fn expansion(tree: &mut Tree<MCTSNode>, to_expand: (NodeId, &MCTSNode)) -> NodeId {
    use id_tree::InsertBehavior::*;

    let (to_expand_id, to_expand_node) = to_expand;
    let move_color = to_expand_node.current_turn.flip_color();

    let moves = compute_moves(&to_expand_node.config, Phase::Moving /* TODO!!! */, move_color);
    let child_node_ids: Vec<NodeId> = moves
        .iter()
        .map(|m| apply_move(&to_expand_node.config, m, move_color))
        .map(|config| {
            tree.insert(
                Node::new(MCTSNode::new(config, Phase::Moving /* TODO!!! */, move_color)),
                UnderNode(&to_expand_id),
            )
            .unwrap()
        })
        .collect();

    let i = unsafe { RNG.generate_range(0..child_node_ids.len()) };
    child_node_ids[i].clone()
}

fn simulation(config: Configuration, color: State, root_color: State) {
    let mut current_config = config;
    let mut current_color = color;

    // While not-terminal-config do:
    let nash = loop {
        let moves = compute_moves(&current_config, Phase::Moving /* TODO */, current_color);
        let i = unsafe { RNG.generate_range(0..moves.len()) };
        current_config = apply_move(&current_config, &moves[i], color);
        current_color = current_color.flip_color();

        /* if is_terminal(current_config) {break calculate_nash(current_config, root_start);} */
    };
}

fn back_propagation(tree: &mut Tree<MCTSNode>, leaf_id: NodeId, nash: i32, leaf_turn: State) {
    let mut current_node_id = leaf_id.clone();
    let mut current_node = tree.get_mut(&leaf_id).unwrap();
    let mut current_turn = leaf_turn;

    loop {
        current_node.data_mut().visitCount += 1;
        if (nash == 1 && current_turn == leaf_turn) || (nash == -1 && current_turn == leaf_turn.flip_color()) {
            current_node.data_mut().winCount += 1;
        }

        (current_node_id, current_node) = if let Ok(ancenstor_ids) = tree.ancestor_ids(&leaf_id) {
            let parent_id = ancenstor_ids.last().unwrap();
            (*parent_id, tree.get_mut(parent_id).unwrap())
        } else {
            break;
        };
        current_turn = current_turn.flip_color();
    }
}
