use miles_control_api::{JobState, LoopPhase};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobAction {
    Start,
    Pause,
    Resume,
    Stop,
}

impl JobAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobAction::Start => "start",
            JobAction::Pause => "pause",
            JobAction::Resume => "resume",
            JobAction::Stop => "stop",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifecycleStep {
    pub state: JobState,
    pub phase: LoopPhase,
    pub message: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleDecision {
    Noop(&'static str),
    Apply(Vec<LifecycleStep>),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LifecycleError {
    #[error("action {action} is invalid from state {state:?}")]
    InvalidTransition {
        state: JobState,
        action: &'static str,
    },
}

pub fn plan_action(
    state: JobState,
    action: JobAction,
) -> Result<LifecycleDecision, LifecycleError> {
    let decision = match action {
        JobAction::Start => match state {
            JobState::Created | JobState::Stopped => LifecycleDecision::Apply(vec![
                LifecycleStep {
                    state: JobState::Starting,
                    phase: LoopPhase::Init,
                    message: "job starting",
                },
                LifecycleStep {
                    state: JobState::Running,
                    phase: LoopPhase::PrepareRollout,
                    message: "job running",
                },
            ]),
            JobState::Starting | JobState::Running | JobState::Paused => {
                LifecycleDecision::Noop("job already started")
            }
            JobState::Stopping | JobState::Failed => {
                return Err(LifecycleError::InvalidTransition {
                    state,
                    action: action.as_str(),
                });
            }
        },
        JobAction::Pause => match state {
            JobState::Running => LifecycleDecision::Apply(vec![LifecycleStep {
                state: JobState::Paused,
                phase: LoopPhase::PrepareRollout,
                message: "job paused",
            }]),
            JobState::Paused => LifecycleDecision::Noop("job already paused"),
            _ => {
                return Err(LifecycleError::InvalidTransition {
                    state,
                    action: action.as_str(),
                });
            }
        },
        JobAction::Resume => match state {
            JobState::Paused => LifecycleDecision::Apply(vec![LifecycleStep {
                state: JobState::Running,
                phase: LoopPhase::PrepareRollout,
                message: "job resumed",
            }]),
            JobState::Running => LifecycleDecision::Noop("job already running"),
            _ => {
                return Err(LifecycleError::InvalidTransition {
                    state,
                    action: action.as_str(),
                });
            }
        },
        JobAction::Stop => match state {
            JobState::Stopped | JobState::Stopping => {
                LifecycleDecision::Noop("job already stopping or stopped")
            }
            JobState::Created
            | JobState::Starting
            | JobState::Running
            | JobState::Paused
            | JobState::Failed => LifecycleDecision::Apply(vec![
                LifecycleStep {
                    state: JobState::Stopping,
                    phase: LoopPhase::Cleanup,
                    message: "job stopping",
                },
                LifecycleStep {
                    state: JobState::Stopped,
                    phase: LoopPhase::Cleanup,
                    message: "job stopped",
                },
            ]),
        },
    };
    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_from_created_transitions_to_running() {
        let decision =
            plan_action(JobState::Created, JobAction::Start).expect("start from created");
        match decision {
            LifecycleDecision::Apply(steps) => {
                assert_eq!(steps.len(), 2);
                assert_eq!(steps[0].state, JobState::Starting);
                assert_eq!(steps[1].state, JobState::Running);
            }
            LifecycleDecision::Noop(_) => panic!("expected transitions"),
        }
    }

    #[test]
    fn start_is_noop_when_running() {
        let decision =
            plan_action(JobState::Running, JobAction::Start).expect("start from running");
        assert_eq!(decision, LifecycleDecision::Noop("job already started"));
    }

    #[test]
    fn pause_and_resume_flow_is_valid() {
        let pause = plan_action(JobState::Running, JobAction::Pause).expect("pause");
        assert!(matches!(pause, LifecycleDecision::Apply(_)));

        let resume = plan_action(JobState::Paused, JobAction::Resume).expect("resume");
        assert!(matches!(resume, LifecycleDecision::Apply(_)));
    }

    #[test]
    fn invalid_transitions_are_rejected() {
        let pause_created =
            plan_action(JobState::Created, JobAction::Pause).expect_err("pause from created");
        assert!(matches!(
            pause_created,
            LifecycleError::InvalidTransition { .. }
        ));

        let resume_created =
            plan_action(JobState::Created, JobAction::Resume).expect_err("resume from created");
        assert!(matches!(
            resume_created,
            LifecycleError::InvalidTransition { .. }
        ));

        let start_failed =
            plan_action(JobState::Failed, JobAction::Start).expect_err("start from failed");
        assert!(matches!(
            start_failed,
            LifecycleError::InvalidTransition { .. }
        ));
    }

    #[test]
    fn stop_is_idempotent_for_stopping_or_stopped() {
        let stopping =
            plan_action(JobState::Stopping, JobAction::Stop).expect("stop from stopping");
        assert_eq!(
            stopping,
            LifecycleDecision::Noop("job already stopping or stopped")
        );

        let stopped = plan_action(JobState::Stopped, JobAction::Stop).expect("stop from stopped");
        assert_eq!(
            stopped,
            LifecycleDecision::Noop("job already stopping or stopped")
        );
    }
}
