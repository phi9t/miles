use miles_control_api::ResourcePlan;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementPlan {
    pub total_gpus: u32,
    pub actor_gpu_indices: Vec<u32>,
    pub critic_gpu_indices: Vec<u32>,
    pub rollout_gpu_indices: Vec<u32>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SchedulerError {
    #[error("resource plan values must be positive")]
    InvalidPlan,
    #[error("rollout gpus exceed total allocated gpus")]
    RolloutOutOfRange,
    #[error("critic gpus exceed total allocated gpus")]
    CriticOutOfRange,
}

pub fn build_placement_plan(
    plan: &ResourcePlan,
    use_critic: bool,
) -> Result<PlacementPlan, SchedulerError> {
    if plan.actor_num_nodes == 0
        || plan.actor_num_gpus_per_node == 0
        || plan.rollout_num_gpus_per_engine == 0
        || plan.num_gpus_per_node == 0
    {
        return Err(SchedulerError::InvalidPlan);
    }

    let actor_total = plan.actor_num_nodes * plan.actor_num_gpus_per_node;
    let critic_total = if use_critic {
        plan.critic_num_nodes.unwrap_or(0) * plan.critic_num_gpus_per_node.unwrap_or(0)
    } else {
        0
    };

    let (total_gpus, rollout_offset, critic_offset) = if plan.colocate {
        let total = actor_total + critic_total;
        (total, 0, actor_total)
    } else {
        let total = actor_total + critic_total + plan.rollout_num_gpus;
        let critic_offset = actor_total;
        let rollout_offset = actor_total + critic_total;
        (total, rollout_offset, critic_offset)
    };

    if !plan.colocate && rollout_offset + plan.rollout_num_gpus > total_gpus {
        return Err(SchedulerError::RolloutOutOfRange);
    }

    if use_critic && critic_offset + critic_total > total_gpus {
        return Err(SchedulerError::CriticOutOfRange);
    }

    let actor_gpu_indices = (0..actor_total).collect::<Vec<_>>();
    let critic_gpu_indices = if use_critic {
        (critic_offset..(critic_offset + critic_total)).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let rollout_gpu_indices = if plan.colocate {
        actor_gpu_indices.clone()
    } else {
        (rollout_offset..(rollout_offset + plan.rollout_num_gpus)).collect::<Vec<_>>()
    };

    Ok(PlacementPlan {
        total_gpus,
        actor_gpu_indices,
        critic_gpu_indices,
        rollout_gpu_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use miles_control_api::ResourcePlan;

    fn base_plan() -> ResourcePlan {
        ResourcePlan {
            actor_num_nodes: 1,
            actor_num_gpus_per_node: 2,
            critic_num_nodes: Some(1),
            critic_num_gpus_per_node: Some(1),
            rollout_num_gpus: 2,
            rollout_num_gpus_per_engine: 1,
            num_gpus_per_node: 8,
            colocate: false,
        }
    }

    #[test]
    fn disaggregated_splits_actor_critic_rollout() {
        let plan = base_plan();
        let p = build_placement_plan(&plan, true).expect("placement should build");
        assert_eq!(p.total_gpus, 5);
        assert_eq!(p.actor_gpu_indices, vec![0, 1]);
        assert_eq!(p.critic_gpu_indices, vec![2]);
        assert_eq!(p.rollout_gpu_indices, vec![3, 4]);
    }

    #[test]
    fn colocate_reuses_actor_gpus_for_rollout() {
        let mut plan = base_plan();
        plan.colocate = true;
        plan.rollout_num_gpus = 2;
        let p = build_placement_plan(&plan, false).expect("placement should build");
        assert_eq!(p.total_gpus, 2);
        assert_eq!(p.actor_gpu_indices, vec![0, 1]);
        assert_eq!(p.rollout_gpu_indices, vec![0, 1]);
    }

    #[test]
    fn no_critic_produces_empty_critic_indices() {
        let plan = base_plan();
        let p = build_placement_plan(&plan, false).expect("placement should build");
        assert_eq!(p.critic_gpu_indices, Vec::<u32>::new());
        assert_eq!(p.total_gpus, 4);
        assert_eq!(p.rollout_gpu_indices, vec![2, 3]);
    }

    #[test]
    fn colocate_with_critic_keeps_rollout_on_actor_indices() {
        let mut plan = base_plan();
        plan.colocate = true;
        plan.rollout_num_gpus = 2;
        let p = build_placement_plan(&plan, true).expect("placement should build");
        assert_eq!(p.total_gpus, 3);
        assert_eq!(p.actor_gpu_indices, vec![0, 1]);
        assert_eq!(p.critic_gpu_indices, vec![2]);
        assert_eq!(p.rollout_gpu_indices, vec![0, 1]);
    }

    #[test]
    fn invalid_plan_rejects_zero_resource_fields() {
        for update in [
            ("actor_num_nodes", 0_u32, 2_u32, 1_u32, 8_u32),
            ("actor_num_gpus_per_node", 1_u32, 0_u32, 1_u32, 8_u32),
            ("rollout_num_gpus_per_engine", 1_u32, 2_u32, 0_u32, 8_u32),
            ("num_gpus_per_node", 1_u32, 2_u32, 1_u32, 0_u32),
        ] {
            let mut plan = base_plan();
            plan.actor_num_nodes = update.1;
            plan.actor_num_gpus_per_node = update.2;
            plan.rollout_num_gpus_per_engine = update.3;
            plan.num_gpus_per_node = update.4;
            let err = build_placement_plan(&plan, false).expect_err("plan should be invalid");
            assert_eq!(err, SchedulerError::InvalidPlan, "case {}", update.0);
        }
    }

    #[test]
    fn deterministic_index_invariants_hold_for_multiple_shapes() {
        let cases = vec![
            ResourcePlan {
                actor_num_nodes: 1,
                actor_num_gpus_per_node: 1,
                critic_num_nodes: None,
                critic_num_gpus_per_node: None,
                rollout_num_gpus: 1,
                rollout_num_gpus_per_engine: 1,
                num_gpus_per_node: 8,
                colocate: false,
            },
            ResourcePlan {
                actor_num_nodes: 2,
                actor_num_gpus_per_node: 2,
                critic_num_nodes: Some(1),
                critic_num_gpus_per_node: Some(2),
                rollout_num_gpus: 4,
                rollout_num_gpus_per_engine: 2,
                num_gpus_per_node: 8,
                colocate: false,
            },
        ];

        for plan in cases {
            let placement = build_placement_plan(&plan, plan.critic_num_nodes.is_some())
                .expect("valid placement");
            assert_eq!(
                placement.actor_gpu_indices,
                (0..(plan.actor_num_nodes * plan.actor_num_gpus_per_node)).collect::<Vec<_>>()
            );
            if !plan.colocate {
                if let (Some(critic_nodes), Some(critic_gpus)) =
                    (plan.critic_num_nodes, plan.critic_num_gpus_per_node)
                {
                    let expected_start = plan.actor_num_nodes * plan.actor_num_gpus_per_node;
                    let expected_len = critic_nodes * critic_gpus;
                    assert_eq!(placement.critic_gpu_indices.len() as u32, expected_len);
                    if expected_len > 0 {
                        assert_eq!(placement.critic_gpu_indices[0], expected_start);
                    }
                }
                assert_eq!(
                    placement.rollout_gpu_indices.len() as u32,
                    plan.rollout_num_gpus
                );
            }
        }
    }
}
