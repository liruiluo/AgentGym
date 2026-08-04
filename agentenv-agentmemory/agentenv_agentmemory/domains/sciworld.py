from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..runtime.domain import DomainContract, DomainTransition


SCIWORLD_DOMAIN_ID = "sciworld"
SCIWORLD_CONDUCTIVITY_SURFACE = "sciworld_conductivity_memory_v1"
SCIWORLD_MELTINGPOINT_SURFACE = "sciworld_meltingpoint_memory_v1"
SCIWORLD_FRICTION_SURFACE = "sciworld_friction_memory_v1"
SCIWORLD_RULE_MEMORY_SURFACE = "sciworld_rule_memory_v1"
SCIWORLD_SOP_MEMORY_SURFACE = "sciworld_sop_memory_v1"
SCIWORLD_NEGATIVE_EVIDENCE_SURFACE = "sciworld_negative_evidence_memory_v1"
SCIWORLD_HYPOTHESIS_TRACKING_SURFACE = "sciworld_hypothesis_tracking_memory_v1"
SCIWORLD_CALIBRATION_SURFACE = "sciworld_calibration_memory_v1"
SCIWORLD_CONTEXTUAL_RULE_SURFACE = "sciworld_contextual_rule_memory_v1"
SCIWORLD_STATE_CHANGE_SURFACE = "sciworld_state_change_memory_v1"
SCIWORLD_GOAL_PROGRESS_SURFACE = "sciworld_goal_progress_memory_v1"
SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE = "sciworld_lab_notebook_longhorizon_v1"
SCIWORLD_SURFACES = {
    "conductivity_memory": SCIWORLD_CONDUCTIVITY_SURFACE,
    "meltingpoint_memory": SCIWORLD_MELTINGPOINT_SURFACE,
    "friction_memory": SCIWORLD_FRICTION_SURFACE,
    "rule_memory": SCIWORLD_RULE_MEMORY_SURFACE,
    "sop_memory": SCIWORLD_SOP_MEMORY_SURFACE,
    "negative_evidence_memory": SCIWORLD_NEGATIVE_EVIDENCE_SURFACE,
    "hypothesis_tracking_memory": SCIWORLD_HYPOTHESIS_TRACKING_SURFACE,
    "calibration_memory": SCIWORLD_CALIBRATION_SURFACE,
    "contextual_rule_memory": SCIWORLD_CONTEXTUAL_RULE_SURFACE,
    "state_change_memory": SCIWORLD_STATE_CHANGE_SURFACE,
    "goal_progress_memory": SCIWORLD_GOAL_PROGRESS_SURFACE,
    "lab_notebook_longhorizon": SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE,
}
SCIWORLD_BACKENDS = ("fixture", "scienceworld")
SCIWORLD_SPLITS = ("train", "dev", "test")

_SINGLE_NATIVE_EPISODE = "single_native_episode"
_SINGLE_CONTINUOUS_LONGHORIZON_EPISODE = "single_continuous_native_episode"
_MULTI_EPISODE_SOP_ORBIT = "multi_native_episode_procedure_transfer_orbit"
_NATIVE_SUCCESS_SCORE = 100.0

_NATIVE_TASK_PLANS = {
    SCIWORLD_CONDUCTIVITY_SURFACE: (
        "test-conductivity-of-unknown-substances",
    ),
    SCIWORLD_SOP_MEMORY_SURFACE: (
        "test-conductivity",
        "test-conductivity-of-unknown-substances",
    ),
}

_NATIVE_SPLIT_MANIFESTS = {
    "test-conductivity": {
        "max_variations": 900,
        "split_counts": {"train": 450, "dev": 225, "test": 225},
        "split_sha256": "6d9689d79ec3fca8831fbc942036f8911fa9f5bd2d0baec2c81715f0493867b2",
    },
    "test-conductivity-of-unknown-substances": {
        "max_variations": 600,
        "split_counts": {"train": 300, "dev": 150, "test": 150},
        "split_sha256": "a89a6a78f0c13d3984f4a7a11574d255e683795acc25c74fd51e900b6b2e55d8",
    },
}

_SCI_ACTION_RE = re.compile(r"\ASCI_ACTION\s+(\{.*\})\Z", flags=re.DOTALL)
_ANSWER_RE = re.compile(r"\AANSWER\s+(\{.*\})\Z", flags=re.DOTALL)


@dataclass(frozen=True)
class SciWorldSurfaceConfig:
    surface: str
    contract_id: str
    memory_kind: str
    system_prompt: str
    max_steps: int
    native_task_family: str


def _surface_configs() -> dict[str, SciWorldSurfaceConfig]:
    common = (
        "You are operating a SciWorld lab through AgentMemoryGym. Run one "
        "executable lab action at a time. Private target facts and future tasks "
        "are never exposed. The environment does not write lab notes, summarize "
        "history, or maintain a helpful rolling transcript for you. Follow this "
        "surface's declared native episode structure; extra session boundaries "
        "must never be invented merely to clear history. As the visible trace "
        "grows, decide when to use SUMMARY/FILTER and what to keep with "
        "ADD/UPDATE/RETRIEVE. If your memory is insufficient, run another visible "
        "experiment instead of inventing a result."
    )
    return {
        SCIWORLD_CONDUCTIVITY_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_CONDUCTIVITY_SURFACE,
            contract_id="sciworld_conductivity_memory_v1_20260803",
            memory_kind="experimental_fact_and_object_property_binding",
            system_prompt=(
                common
                + " This surface focuses on remembering experimental conductivity "
                "results for unknown materials and using them in later phases."
            ),
            max_steps=64,
            native_task_family="test-conductivity-of-unknown-substances",
        ),
        SCIWORLD_MELTINGPOINT_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_MELTINGPOINT_SURFACE,
            contract_id="sciworld_meltingpoint_memory_v1_20260803",
            memory_kind="numeric_experimental_memory",
            system_prompt=(
                common
                + " This surface focuses on remembering measured melting points "
                "and using exact numeric thresholds later."
            ),
            max_steps=72,
            native_task_family="measure-melting-point-unknown-substance",
        ),
        SCIWORLD_FRICTION_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_FRICTION_SURFACE,
            contract_id="sciworld_friction_memory_v1_20260803",
            memory_kind="comparative_experimental_memory",
            system_prompt=(
                common
                + " This surface focuses on remembering comparative friction "
                "measurements for unnamed surfaces and using the ranking later."
            ),
            max_steps=72,
            native_task_family="inclined-plane-friction-unnamed-surfaces",
        ),
        SCIWORLD_RULE_MEMORY_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_RULE_MEMORY_SURFACE,
            contract_id="sciworld_rule_memory_v1_20260803",
            memory_kind="multi_experiment_fact_rule_induction",
            system_prompt=(
                common
                + " This surface focuses on deriving a reusable scientific fact "
                "or rule from multiple experiments. It is fact/rule memory, not "
                "procedure memory."
            ),
            max_steps=96,
            native_task_family="chemistry-mix-paint-secondary-and-tertiary-color",
        ),
        SCIWORLD_SOP_MEMORY_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_SOP_MEMORY_SURFACE,
            contract_id="sciworld_sop_memory_v1_20260803",
            memory_kind="procedural_sop_memory",
            system_prompt=(
                common
                + " This surface focuses on remembering reusable lab procedure, "
                "such as how to test conductivity, rather than remembering only "
                "the measured fact. It intentionally spans semantically distinct "
                "native episodes or tasks: per-episode trace resets at a real task "
                "boundary, while the policy-authored external notebook persists."
            ),
            max_steps=96,
            native_task_family="procedure-transfer-across-sciworld-tasks",
        ),
        SCIWORLD_NEGATIVE_EVIDENCE_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_NEGATIVE_EVIDENCE_SURFACE,
            contract_id="sciworld_negative_evidence_memory_v1_20260803",
            memory_kind="negative_experimental_evidence",
            system_prompt=(
                common
                + " This surface focuses on remembering failed/null experiment "
                "results and using them to exclude a candidate later, rather "
                "than repeating the same dead-end."
            ),
            max_steps=96,
            native_task_family="negative-evidence-and-elimination",
        ),
        SCIWORLD_HYPOTHESIS_TRACKING_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_HYPOTHESIS_TRACKING_SURFACE,
            contract_id="sciworld_hypothesis_tracking_memory_v1_20260803",
            memory_kind="hypothesis_and_evidence_tracking",
            system_prompt=(
                common
                + " This surface focuses on tracking competing hypotheses and "
                "which experiment supported or ruled out each hypothesis."
            ),
            max_steps=128,
            native_task_family="multi-experiment-hypothesis-tracking",
        ),
        SCIWORLD_CALIBRATION_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_CALIBRATION_SURFACE,
            contract_id="sciworld_calibration_memory_v1_20260803",
            memory_kind="instrument_calibration_memory",
            system_prompt=(
                common
                + " This surface focuses on remembering an instrument calibration "
                "or measurement offset and applying it to later observations."
            ),
            max_steps=96,
            native_task_family="instrument-calibration-and-corrected-measurement",
        ),
        SCIWORLD_CONTEXTUAL_RULE_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_CONTEXTUAL_RULE_SURFACE,
            contract_id="sciworld_contextual_rule_memory_v1_20260803",
            memory_kind="conditioned_scientific_rule_memory",
            system_prompt=(
                common
                + " This surface focuses on remembering that a scientific rule is "
                "conditional on the experimental context, not universally true."
            ),
            max_steps=128,
            native_task_family="context-dependent-scientific-rule",
        ),
        SCIWORLD_STATE_CHANGE_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_STATE_CHANGE_SURFACE,
            contract_id="sciworld_state_change_memory_v1_20260803",
            memory_kind="memory_revision_after_new_evidence",
            system_prompt=(
                common
                + " This surface focuses on updating or replacing an earlier lab "
                "note when a later experiment shows that the old state is no "
                "longer the right one to use."
            ),
            max_steps=128,
            native_task_family="state-change-and-memory-revision",
        ),
        SCIWORLD_GOAL_PROGRESS_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_GOAL_PROGRESS_SURFACE,
            contract_id="sciworld_goal_progress_memory_v1_20260803",
            memory_kind="unfinished_goal_progress_memory",
            system_prompt=(
                common
                + " This surface focuses on remembering unfinished experiment "
                "progress, completed subgoals, and the next step after a phase "
                "boundary."
            ),
            max_steps=128,
            native_task_family="multi-step-goal-progress-tracking",
        ),
        SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE: SciWorldSurfaceConfig(
            surface=SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE,
            contract_id="sciworld_lab_notebook_longhorizon_v1_20260803",
            memory_kind="self_managed_external_lab_notebook",
            system_prompt=(
                common
                + " This long-horizon surface remains one continuous native episode "
                "and is intended to exceed raw-context comfort unless the policy "
                "compresses its own visible trace and maintains an external lab "
                "notebook with policy-authored memory actions."
            ),
            max_steps=512,
            native_task_family="multi-experiment-lab-notebook-chain",
        ),
    }


SCIWORLD_SURFACE_CONFIGS = _surface_configs()


def _formal_episode_contract(surface: str) -> tuple[str, str]:
    if surface == SCIWORLD_SOP_MEMORY_SURFACE:
        return (
            _MULTI_EPISODE_SOP_ORBIT,
            "required_native_episode_boundaries_preserve_ltm_reset_local_trace",
        )
    if surface == SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE:
        return (
            _SINGLE_CONTINUOUS_LONGHORIZON_EPISODE,
            "no_boundary_before_native_episode_terminal",
        )
    return _SINGLE_NATIVE_EPISODE, "native_episode_boundaries_only"


def contract_for_surface(surface: str) -> DomainContract:
    try:
        config = SCIWORLD_SURFACE_CONFIGS[surface]
    except KeyError as exc:
        raise ValueError(f"unsupported SciWorld surface: {surface!r}") from exc
    return DomainContract(
        contract_id=config.contract_id,
        system_prompt=config.system_prompt,
        native_action_descriptions=(
            'SCI_ACTION {"action": "<native SciWorld text command>"}',
            'ANSWER {"answer": "..."}',
        ),
        max_steps=config.max_steps,
    )


@dataclass(frozen=True)
class SciWorldFixturePhase:
    observation: str
    experiment_keywords: tuple[str, ...]
    experiment_result: str
    answer_keywords: tuple[str, ...]
    dependency: str


@dataclass(frozen=True)
class SciWorldFixtureTask:
    task_id: str
    phases: tuple[SciWorldFixturePhase, ...]

    @property
    def phase_count(self) -> int:
        return len(self.phases)


def _fixture_tasks_for_surface(surface: str) -> tuple[SciWorldFixtureTask, ...]:
    if surface == SCIWORLD_CONDUCTIVITY_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="conductivity_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "Conductivity source phase. The lab has unknown sample "
                            "alpha. Determine whether it conducts electricity."
                        ),
                        experiment_keywords=("conduct",),
                        experiment_result=(
                            "You build a simple circuit. Result: unknown sample "
                            "alpha is conductive. The lab result is visible now "
                            "but is not repeated after the phase changes."
                        ),
                        answer_keywords=("conductive",),
                        dependency="measure_conductivity",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Conductivity dependent phase. Build a working circuit "
                            "using one material. Candidate materials: unknown sample "
                            "alpha; rubber strip beta. The prior lab result is not "
                            "repeated in this observation."
                        ),
                        experiment_keywords=("conduct", "alpha"),
                        experiment_result=(
                            "A fresh circuit test shows unknown sample alpha conducts."
                        ),
                        answer_keywords=("unknown sample alpha",),
                        dependency="use_prior_conductivity_result",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_MELTINGPOINT_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="meltingpoint_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "Melting-point source phase. The lab has unknown crystal "
                            "mira. Measure its melting point."
                        ),
                        experiment_keywords=("melt",),
                        experiment_result=(
                            "You heat the sample slowly. Result: unknown crystal "
                            "mira melts at 70 celsius."
                        ),
                        answer_keywords=("70",),
                        dependency="measure_numeric_melting_point",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Melting-point dependent phase. Choose the material that "
                            "will melt below 80 celsius. Candidates: unknown crystal "
                            "mira; ceramic bead nora. The previous measurement is not "
                            "shown again."
                        ),
                        experiment_keywords=("melt", "mira"),
                        experiment_result="A fresh measurement shows mira melts at 70 celsius.",
                        answer_keywords=("mira",),
                        dependency="use_numeric_threshold_memory",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_FRICTION_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="friction_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "Friction source phase. Two unnamed surfaces, slate A "
                            "and glass B, are available. Measure which has greater "
                            "friction on the inclined plane."
                        ),
                        experiment_keywords=("friction",),
                        experiment_result=(
                            "The cart rolls less far on slate A. Result: slate A has "
                            "higher friction than glass B."
                        ),
                        answer_keywords=("slate", "higher"),
                        dependency="measure_relative_friction",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Friction dependent phase. Choose a surface to slow a cart. "
                            "Candidates: slate A; glass B. The earlier ranking is not "
                            "shown again."
                        ),
                        experiment_keywords=("friction", "slate"),
                        experiment_result="A fresh test again shows slate A has higher friction.",
                        answer_keywords=("slate",),
                        dependency="use_prior_friction_ranking",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_RULE_MEMORY_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="rule_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation="Rule source phase 1. Mix red paint with yellow paint.",
                        experiment_keywords=("red", "yellow"),
                        experiment_result="The mixture becomes orange.",
                        answer_keywords=("orange",),
                        dependency="observe_color_rule_part_1",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Rule source phase 2. Mix orange paint with yellow paint. "
                            "The previous red/yellow result is not repeated."
                        ),
                        experiment_keywords=("orange", "yellow"),
                        experiment_result="The mixture becomes amber.",
                        answer_keywords=("amber",),
                        dependency="observe_color_rule_part_2",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Rule dependent phase. A later task needs amber paint, "
                            "starting only from the available primary paints red and "
                            "yellow. State the two-step mixing plan. Prior experiments "
                            "are not repeated."
                        ),
                        experiment_keywords=("amber",),
                        experiment_result=(
                            "If you rerun the experiment, orange plus yellow makes amber."
                        ),
                        answer_keywords=("red", "yellow", "orange"),
                        dependency="use_induced_color_rule",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_SOP_MEMORY_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="sop_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "SOP source phase. Learn the procedure for testing whether "
                            "a material conducts electricity."
                        ),
                        experiment_keywords=("assemble", "circuit"),
                        experiment_result=(
                            "Procedure result: connect battery, wire, bulb, and the "
                            "sample in one circuit; a lit bulb means the sample conducts."
                        ),
                        answer_keywords=("battery", "bulb", "sample"),
                        dependency="learn_conductivity_test_procedure",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "SOP dependent phase. A new unknown material must be tested, "
                            "but the lab manual is not shown again. State the procedure "
                            "you should perform."
                        ),
                        experiment_keywords=("assemble", "circuit"),
                        experiment_result=(
                            "Rebuilding the procedure: battery, wire, bulb, sample; "
                            "observe whether the bulb lights."
                        ),
                        answer_keywords=("battery", "bulb", "sample"),
                        dependency="reuse_procedure_not_fact",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_NEGATIVE_EVIDENCE_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="negative_evidence_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "Negative-evidence source phase. Test whether powder "
                            "zeta fizzes when vinegar is added."
                        ),
                        experiment_keywords=("vinegar", "zeta"),
                        experiment_result=(
                            "You add vinegar to powder zeta. Result: powder zeta "
                            "does not fizz with vinegar."
                        ),
                        answer_keywords=("does not fizz",),
                        dependency="observe_null_reaction",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Negative-evidence dependent phase. Exactly one candidate "
                            "should fizz with vinegar. Candidates: powder zeta; powder "
                            "eta. The earlier failed test is not repeated."
                        ),
                        experiment_keywords=("vinegar", "eta"),
                        experiment_result="A fresh test shows powder eta fizzes with vinegar.",
                        answer_keywords=("powder eta",),
                        dependency="exclude_candidate_from_negative_evidence",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_HYPOTHESIS_TRACKING_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="hypothesis_tracking_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "Hypothesis source phase 1. A sprout leans after the lamp "
                            "is moved to the east side. Record which hypothesis this "
                            "supports."
                        ),
                        experiment_keywords=("lamp", "east"),
                        experiment_result=(
                            "The sprout bends east toward the lamp. Evidence supports "
                            "the light-direction hypothesis."
                        ),
                        answer_keywords=("light",),
                        dependency="support_first_hypothesis",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Hypothesis source phase 2. Keep the lamp fixed but swap "
                            "the soil tray color. Record whether the color hypothesis "
                            "is supported."
                        ),
                        experiment_keywords=("soil", "color"),
                        experiment_result=(
                            "The sprout direction does not change when only soil color "
                            "changes. Evidence rules out the soil-color hypothesis."
                        ),
                        answer_keywords=("rules out", "soil"),
                        dependency="rule_out_competing_hypothesis",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Hypothesis dependent phase. Choose the supported cause of "
                            "the sprout direction. Prior evidence is not repeated."
                        ),
                        experiment_keywords=("lamp", "soil"),
                        experiment_result=(
                            "Rerunning the paired tests again supports light direction "
                            "and rules out soil color."
                        ),
                        answer_keywords=("light direction",),
                        dependency="retrieve_supported_hypothesis",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_CALIBRATION_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="calibration_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "Calibration source phase. Compare thermometer T with a "
                            "reference bath known to be 50 celsius."
                        ),
                        experiment_keywords=("thermometer", "reference"),
                        experiment_result=(
                            "Thermometer T reads 55 celsius in the 50 celsius bath. "
                            "Result: thermometer T reads 5 celsius high."
                        ),
                        answer_keywords=("5", "high"),
                        dependency="measure_instrument_offset",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Calibration dependent phase. Thermometer T now reads 75 "
                            "celsius for sample mira. Report the corrected temperature. "
                            "The calibration result is not repeated."
                        ),
                        experiment_keywords=("thermometer", "mira"),
                        experiment_result=(
                            "The raw thermometer reading is 75 celsius; applying the "
                            "stored offset gives 70 celsius."
                        ),
                        answer_keywords=("70",),
                        dependency="apply_calibration_memory",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_CONTEXTUAL_RULE_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="contextual_rule_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "Contextual-rule source phase. Compare how sugar dissolves "
                            "in cold water and hot water."
                        ),
                        experiment_keywords=("sugar", "water"),
                        experiment_result=(
                            "Sugar dissolves slowly in cold water but quickly in hot "
                            "water. The speed rule depends on water temperature."
                        ),
                        answer_keywords=("hot", "quickly"),
                        dependency="observe_conditioned_rule",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Contextual-rule dependent phase. You need sugar to dissolve "
                            "quickly. Choose the context to use. Prior experiments are "
                            "not repeated."
                        ),
                        experiment_keywords=("sugar", "quickly"),
                        experiment_result=(
                            "Repeating the experiment shows hot water is the fast "
                            "dissolving context."
                        ),
                        answer_keywords=("hot water",),
                        dependency="apply_conditioned_rule",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_STATE_CHANGE_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="state_change_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "State-change source phase. A quick indicator strip is used "
                            "on solution riva."
                        ),
                        experiment_keywords=("indicator", "riva"),
                        experiment_result=(
                            "The quick strip suggests solution riva is acidic. This is "
                            "a preliminary result."
                        ),
                        answer_keywords=("acidic",),
                        dependency="record_preliminary_state",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "State-change revision phase. A calibrated pH meter is now "
                            "used on solution riva."
                        ),
                        experiment_keywords=("meter", "riva"),
                        experiment_result=(
                            "The calibrated pH meter shows solution riva is neutral. "
                            "This supersedes the preliminary strip result."
                        ),
                        answer_keywords=("neutral",),
                        dependency="revise_state_after_new_evidence",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "State-change dependent phase. Classify solution riva using "
                            "the latest reliable evidence. Prior measurements are not "
                            "repeated."
                        ),
                        experiment_keywords=("classify", "riva"),
                        experiment_result=(
                            "The latest reliable evidence is the calibrated pH meter: "
                            "riva is neutral."
                        ),
                        answer_keywords=("neutral",),
                        dependency="use_revised_state",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_GOAL_PROGRESS_SURFACE:
        return (
            SciWorldFixtureTask(
                task_id="goal_progress_fixture_000",
                phases=(
                    SciWorldFixturePhase(
                        observation=(
                            "Goal-progress phase 1. The experiment plan has three "
                            "subgoals: collect sample, heat sample, then record color. "
                            "Complete the collection subgoal."
                        ),
                        experiment_keywords=("collect", "sample"),
                        experiment_result=(
                            "Subgoal complete: sample collected. Remaining subgoals: "
                            "heat sample, then record color."
                        ),
                        answer_keywords=("sample collected",),
                        dependency="remember_completed_subgoal",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Goal-progress phase 2. Continue the same experiment after "
                            "a phase boundary. The previous progress list is not repeated."
                        ),
                        experiment_keywords=("heat", "sample"),
                        experiment_result=(
                            "Subgoal complete: sample heated. Remaining subgoal: record "
                            "the final color."
                        ),
                        answer_keywords=("sample heated",),
                        dependency="resume_next_subgoal",
                    ),
                    SciWorldFixturePhase(
                        observation=(
                            "Goal-progress final phase. State the only unfinished "
                            "subgoal. Prior progress notes are not repeated."
                        ),
                        experiment_keywords=("record", "color"),
                        experiment_result="The unfinished subgoal is to record final color.",
                        answer_keywords=("record final color",),
                        dependency="retrieve_unfinished_subgoal",
                    ),
                ),
            ),
        )
    if surface == SCIWORLD_LAB_NOTEBOOK_LONGHORIZON_SURFACE:
        colors = (
            ("station 1", "blue"),
            ("station 2", "green"),
            ("station 3", "orange"),
            ("station 4", "silver"),
            ("station 5", "violet"),
            ("station 6", "black"),
        )
        phases = []
        for station, color in colors:
            phases.append(
                SciWorldFixturePhase(
                    observation=(
                        f"Long-horizon notebook phase. Inspect {station} and record "
                        "its sealed indicator color for later. Earlier station "
                        "readings are not repeated."
                    ),
                    experiment_keywords=("inspect", station.split()[1]),
                    experiment_result=f"{station} indicator color is {color}.",
                    answer_keywords=(color,),
                    dependency="record_lab_notebook_entry",
                )
            )
        phases.append(
            SciWorldFixturePhase(
                observation=(
                    "Long-horizon final phase. Which station had the violet indicator? "
                    "The station readings are not repeated; use your own notebook/memory."
                ),
                experiment_keywords=("inspect", "5"),
                experiment_result="A fresh inspection shows station 5 is violet.",
                answer_keywords=("station 5",),
                dependency="retrieve_external_lab_notebook_entry",
            )
        )
        return (SciWorldFixtureTask(task_id="longhorizon_fixture_000", phases=tuple(phases)),)
    raise ValueError(f"unsupported SciWorld fixture surface: {surface!r}")


class SciWorldMemoryFactory:
    domain_id = SCIWORLD_DOMAIN_ID

    def __init__(
        self,
        *,
        surface: str = SCIWORLD_CONDUCTIVITY_SURFACE,
        backend: str = "scienceworld",
        split: str = "train",
        task_count: int | None = None,
    ) -> None:
        if surface not in SCIWORLD_SURFACE_CONFIGS:
            raise ValueError(f"unsupported SciWorld surface: {surface!r}")
        if backend not in SCIWORLD_BACKENDS:
            raise ValueError(
                "SciWorld backend must be one of: " + ", ".join(SCIWORLD_BACKENDS)
            )
        if split not in SCIWORLD_SPLITS:
            raise ValueError(
                "SciWorld split must be one of: " + ", ".join(SCIWORLD_SPLITS)
            )
        self._scienceworld_module = None
        if backend == "scienceworld":
            if surface not in _NATIVE_TASK_PLANS:
                raise RuntimeError(
                    f"SciWorld surface {surface!r} has no frozen native task plan; "
                    "use the fixture backend only for plumbing until an official "
                    "task mapping and split manifest are certified."
                )
            self._scienceworld_module = _require_scienceworld_dependency()
        self.surface = surface
        self.backend = backend
        self.split = split
        self.contract = contract_for_surface(surface)
        self.fixture_tasks = _fixture_tasks_for_surface(surface)
        self.native_task_plan = _NATIVE_TASK_PLANS.get(surface, ())
        if backend == "fixture":
            available_task_count = len(self.fixture_tasks)
        else:
            available_task_count = min(
                int(_NATIVE_SPLIT_MANIFESTS[task]["split_counts"][split])
                for task in self.native_task_plan
            )
        if task_count is not None and not 1 <= task_count <= available_task_count:
            raise ValueError(
                "SciWorld task_count must be between 1 and the frozen split size "
                f"{available_task_count}; got {task_count}."
            )
        self._task_count = task_count or available_task_count

    @property
    def task_count(self) -> int:
        return self._task_count

    def create(self, env_uid: str):
        if self.backend == "fixture":
            return SciWorldFixtureDriver(
                env_uid=env_uid,
                surface=self.surface,
                contract=self.contract,
                fixture_tasks=self.fixture_tasks,
            )
        return ScienceWorldNativeDriver(
            env_uid=env_uid,
            surface=self.surface,
            contract=self.contract,
            module=self._scienceworld_module,
            split=self.split,
            task_names=self.native_task_plan,
        )

    def metadata(self) -> dict[str, Any]:
        config = SCIWORLD_SURFACE_CONFIGS[self.surface]
        formal_episode_structure, session_boundary_policy = _formal_episode_contract(
            self.surface
        )
        episode_structure = (
            formal_episode_structure
            if self.backend == "scienceworld"
            else "fixture_stages_only_not_capability_evidence"
        )
        return {
            "source": "allenai/ScienceWorld",
            "domain_family": "scientific_experiment_lab",
            "backend": self.backend,
            "split": self.split,
            "memory_kind": config.memory_kind,
            "native_task_family": config.native_task_family,
            "native_task_names": list(self.native_task_plan),
            "memory_management": "policy_managed_external_notebook",
            "history_policy": "no_harness_recent_n_no_environment_summary",
            "episode_structure": episode_structure,
            "formal_episode_structure": formal_episode_structure,
            "session_boundary_policy": session_boundary_policy,
            "requires_multi_episode_orchestrator": (
                self.surface == SCIWORLD_SOP_MEMORY_SURFACE
            ),
            "native_split_manifest": {
                task: dict(_NATIVE_SPLIT_MANIFESTS[task])
                for task in self.native_task_plan
            },
            "artificial_session_boundaries": False,
            "context_compaction_owner": "policy",
            "harness_summarizes_history": False,
            "manual_recent_n_window": None,
            "requires_scienceworld_dependency": self.backend == "scienceworld",
            "surfaces": dict(SCIWORLD_SURFACES),
        }


class SciWorldFixtureDriver:
    domain_id = SCIWORLD_DOMAIN_ID

    def __init__(
        self,
        *,
        env_uid: str,
        surface: str,
        contract: DomainContract,
        fixture_tasks: tuple[SciWorldFixtureTask, ...],
    ) -> None:
        self.env_uid = env_uid
        self.surface = surface
        self.contract = contract
        self.fixture_tasks = fixture_tasks
        self.task = fixture_tasks[0]
        self.phase_index = 0
        self.closed = False

    def reset(self, data_idx: int) -> DomainTransition:
        self.task = self.fixture_tasks[int(data_idx) % len(self.fixture_tasks)]
        self.phase_index = 0
        self.closed = False
        return self._transition(self._active_phase().observation, status="active")

    def step(self, action: str, env_step: int) -> DomainTransition:
        if self.closed:
            raise RuntimeError("SciWorld fixture driver is closed")
        try:
            parsed = _parse_native_action(action)
        except ValueError as exc:
            return self._invalid(env_step, action, str(exc))
        if parsed is None:
            return self._invalid(env_step, action, "expected SCI_ACTION or ANSWER")
        op, payload = parsed
        if op == "SCI_ACTION":
            return self._step_sci_action(payload, env_step, action)
        return self._step_answer(payload, env_step, action)

    def close(self) -> None:
        self.closed = True

    def _active_phase(self) -> SciWorldFixturePhase:
        return self.task.phases[min(self.phase_index, self.task.phase_count - 1)]

    def _step_sci_action(
        self,
        payload: dict[str, Any],
        env_step: int,
        raw_action: str,
    ) -> DomainTransition:
        del raw_action
        command = _require_payload_text(payload, "action")
        lowered = command.lower()
        phase = self._active_phase()
        matches = all(keyword.lower() in lowered for keyword in phase.experiment_keywords)
        observation = (
            phase.experiment_result
            if matches
            else "The lab action executes, but it does not resolve the current memory-dependent question."
        )
        return self._transition(
            observation,
            action_execution={
                "op": "SCI_ACTION",
                "status": "executed",
                "step": env_step,
                "command": command,
                "experiment_matched": matches,
            },
            tool_ops=(
                {
                    "op": "SCI_ACTION",
                    "step": env_step,
                    "command": command,
                    "experiment_matched": matches,
                },
            ),
            reward_components=(
                {
                    "name": "sciworld_experiment_observed",
                    "value": 0.0,
                    "op": "SCI_ACTION",
                    "step": env_step,
                    "experiment_matched": matches,
                },
            ),
            domain_evidence={
                "task_id": self.task.task_id,
                "dependency": phase.dependency,
                "experiment_observed": matches,
            },
        )

    def _step_answer(
        self,
        payload: dict[str, Any],
        env_step: int,
        raw_action: str,
    ) -> DomainTransition:
        answer = _require_payload_text(payload, "answer").lower()
        phase = self._active_phase()
        correct = all(keyword.lower() in answer for keyword in phase.answer_keywords)
        if not correct:
            return self._terminal_failure(env_step, raw_action, answer)
        final_phase = self.phase_index >= self.task.phase_count - 1
        if final_phase:
            self.phase_index = self.task.phase_count
            return self._transition(
                "SciWorld memory task complete.",
                reward=1.0,
                done=True,
                status="success",
                episode_success=True,
                action_execution={
                    "op": "ANSWER",
                    "status": "correct",
                    "step": env_step,
                    "answer": answer,
                },
                tool_ops=(
                    {
                        "op": "ANSWER",
                        "step": env_step,
                        "answer": answer,
                        "correct": True,
                    },
                ),
                reward_components=(
                    {
                        "name": "sciworld_final_answer_correct",
                        "value": 1.0,
                        "op": "ANSWER",
                        "step": env_step,
                    },
                ),
                domain_evidence={
                    "task_id": self.task.task_id,
                    "dependency": phase.dependency,
                },
            )
        self.phase_index += 1
        next_phase = self._active_phase()
        return self._transition(
            next_phase.observation,
            reward=1.0,
            status="active",
            action_execution={
                "op": "ANSWER",
                "status": "correct",
                "step": env_step,
                "answer": answer,
                "phase_advanced": True,
            },
            tool_ops=(
                {
                    "op": "ANSWER",
                    "step": env_step,
                    "answer": answer,
                    "correct": True,
                },
            ),
            reward_components=(
                {
                    "name": "sciworld_phase_answer_correct",
                    "value": 1.0,
                    "op": "ANSWER",
                    "step": env_step,
                },
            ),
            domain_evidence={
                "task_id": self.task.task_id,
                "dependency": phase.dependency,
                "phase_advanced": True,
            },
        )

    def _invalid(self, env_step: int, raw_action: str, reason: str) -> DomainTransition:
        return self._transition(
            f"Invalid SciWorld action: {reason}.",
            reward=-0.01,
            action_execution={
                "op": "INVALID",
                "status": "invalid",
                "step": env_step,
                "submitted_action": raw_action,
                "error": reason,
            },
            reward_components=(
                {
                    "name": "invalid_action",
                    "value": -0.01,
                    "op": "INVALID",
                    "step": env_step,
                },
            ),
        )

    def _terminal_failure(
        self, env_step: int, raw_action: str, answer: str) -> DomainTransition:
        return self._transition(
            "Incorrect answer. Episode terminated without revealing the target.",
            reward=0.0,
            done=True,
            status="failed",
            episode_success=False,
            action_execution={
                "op": "ANSWER",
                "status": "incorrect",
                "step": env_step,
                "submitted_action": raw_action,
                "answer": answer,
            },
            reward_components=(
                {
                    "name": "sciworld_answer_incorrect",
                    "value": 0.0,
                    "op": "ANSWER",
                    "step": env_step,
                },
            ),
            domain_evidence={"task_id": self.task.task_id},
        )

    def _transition(
        self,
        observation: str,
        *,
        reward: float = 0.0,
        done: bool = False,
        status: str = "active",
        episode_success: bool = False,
        action_execution=None,
        tool_ops=(),
        reward_components=(),
        domain_evidence=None,
    ) -> DomainTransition:
        evidence = {
            "task_id": self.task.task_id,
            "surface": self.surface,
            "backend": "fixture",
            "memory_kind": SCIWORLD_SURFACE_CONFIGS[self.surface].memory_kind,
            "history_policy": "no_harness_recent_n_no_environment_summary",
        }
        if domain_evidence:
            evidence.update(domain_evidence)
        return DomainTransition(
            observation=(
                f"SciWorld task {self.task.task_id}. Phase "
                f"{min(self.phase_index, self.task.phase_count)}/{self.task.phase_count}. "
                + observation
            ),
            reward=reward,
            done=done,
            status=status,
            phase_index=min(self.phase_index, self.task.phase_count),
            phase_count=self.task.phase_count,
            episode_success=episode_success,
            action_execution=action_execution or {},
            tool_ops=tool_ops,
            reward_components=reward_components,
            domain_evidence=evidence,
        )


class ScienceWorldNativeDriver:
    domain_id = SCIWORLD_DOMAIN_ID

    def __init__(
        self,
        *,
        env_uid: str,
        surface: str,
        contract: DomainContract,
        module,
        split: str,
        task_names: tuple[str, ...],
    ) -> None:
        if module is None:
            raise RuntimeError("ScienceWorld native module was not initialized")
        if not task_names:
            raise RuntimeError("ScienceWorld native driver requires a frozen task plan")
        self.env_uid = env_uid
        self.surface = surface
        self.contract = contract
        self.split = split
        self.task_names = task_names
        self.env = module.ScienceWorldEnv()
        self.closed = False
        self.phase_index = 0
        self.data_idx = 0
        self.task_name = ""
        self.variation_idx = 0
        self._splits_by_task: dict[str, dict[str, tuple[int, ...]]] = {}

    def reset(self, data_idx: int) -> DomainTransition:
        self.closed = False
        self.phase_index = 0
        self.data_idx = int(data_idx)
        return self._load_current_native_episode()

    def step(self, action: str, env_step: int) -> DomainTransition:
        try:
            parsed = _parse_native_action(action)
        except ValueError as exc:
            parsed = None
            parse_error = str(exc)
        else:
            parse_error = "expected SCI_ACTION JSON"
        if parsed is None or parsed[0] != "SCI_ACTION":
            return self._invalid(action, env_step, parse_error)

        command = _require_payload_text(parsed[1], "action")
        completed_task = self.task_name
        completed_variation = self.variation_idx
        observation, reward, native_done, raw_info = self.env.step(command)
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        reward_delta = float(reward)
        score = float(info.get("score", 0.0))
        strict_success = bool(native_done and score >= _NATIVE_SUCCESS_SCORE)
        action_execution = {
            "op": "SCI_ACTION",
            "status": "executed",
            "step": env_step,
            "command": command,
            "native_task_name": completed_task,
            "native_variation_idx": completed_variation,
        }
        tool_ops = (
            {
                "op": "SCI_ACTION",
                "step": env_step,
                "command": command,
                "native_task_name": completed_task,
            },
        )
        reward_components = (
            {
                "name": "scienceworld_reward_delta",
                "value": reward_delta,
                "op": "SCI_ACTION",
                "step": env_step,
            },
        )
        evidence = {
            "scienceworld_info": info,
            "native_score": score,
            "native_done": bool(native_done),
            "native_strict_success": strict_success,
        }

        if not native_done:
            return self._transition(
                observation,
                reward=reward_delta,
                action_execution=action_execution,
                tool_ops=tool_ops,
                reward_components=reward_components,
                domain_evidence=evidence,
            )

        if not strict_success:
            action_execution.update(
                {"status": "failed", "native_episode_completed": True}
            )
            return self._transition(
                observation,
                reward=reward_delta,
                done=True,
                status="failed",
                episode_success=False,
                action_execution=action_execution,
                tool_ops=tool_ops,
                reward_components=reward_components,
                domain_evidence=evidence,
            )

        self.phase_index += 1
        action_execution["native_episode_completed"] = True
        if self.phase_index < len(self.task_names):
            next_transition = self._load_current_native_episode()
            action_execution.update(
                {
                    "phase_advanced": True,
                    "next_native_task_name": self.task_name,
                    "next_native_variation_idx": self.variation_idx,
                }
            )
            evidence.update(
                {
                    "completed_native_task_name": completed_task,
                    "completed_native_variation_idx": completed_variation,
                    "next_scienceworld_info": next_transition.domain_evidence.get(
                        "scienceworld_info", {}
                    ),
                }
            )
            return self._transition(
                next_transition.observation,
                reward=reward_delta,
                status="active",
                action_execution=action_execution,
                tool_ops=tool_ops,
                reward_components=reward_components,
                domain_evidence=evidence,
            )

        action_execution["status"] = "success"
        return self._transition(
            observation,
            reward=reward_delta,
            done=True,
            status="success",
            episode_success=True,
            action_execution=action_execution,
            tool_ops=tool_ops,
            reward_components=reward_components,
            domain_evidence=evidence,
        )

    def close(self) -> None:
        if not self.closed:
            self.env.close()
            self.closed = True

    def _load_current_native_episode(self) -> DomainTransition:
        task_name = self.task_names[self.phase_index]
        variation_idx = self._variation_for(task_name, self.data_idx)
        self.env.load(task_name, variation_idx)
        observation, raw_info = self.env.reset()
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        self.task_name = task_name
        self.variation_idx = variation_idx
        task_description = str(
            info.get("taskDesc") or self.env.get_task_description()
        ).strip()
        visible_observation = "\n\n".join(
            part for part in (task_description, str(observation).strip()) if part
        )
        return self._transition(
            visible_observation,
            status="active",
            domain_evidence={
                "scienceworld_info": info,
                "native_episode_index": self.phase_index,
            },
        )

    def _variation_for(self, task_name: str, data_idx: int) -> int:
        if task_name not in self._splits_by_task:
            self.env.load(task_name, 0)
            observed_splits = {
                "train": tuple(int(item) for item in self.env.get_variations_train()),
                "dev": tuple(int(item) for item in self.env.get_variations_dev()),
                "test": tuple(int(item) for item in self.env.get_variations_test()),
            }
            _attest_native_splits(self.env, task_name, observed_splits)
            self._splits_by_task[task_name] = observed_splits
        variations = self._splits_by_task[task_name][self.split]
        return variations[data_idx % len(variations)]

    def _invalid(self, action: str, env_step: int, reason: str) -> DomainTransition:
        return self._transition(
            f"Native ScienceWorld backend expects SCI_ACTION JSON: {reason}.",
            reward=-0.01,
            status="active",
            action_execution={
                "op": "INVALID",
                "status": "invalid",
                "step": env_step,
                "submitted_action": action,
            },
            reward_components=(
                {
                    "name": "invalid_action",
                    "value": -0.01,
                    "op": "INVALID",
                    "step": env_step,
                },
            ),
        )

    def _transition(self, observation: str, **kwargs) -> DomainTransition:
        evidence = {
            "surface": self.surface,
            "backend": "scienceworld",
            "memory_kind": SCIWORLD_SURFACE_CONFIGS[self.surface].memory_kind,
            "task_name": self.task_name,
            "variation_idx": self.variation_idx,
            "split": self.split,
            "native_episode_index": self.phase_index,
            "native_episode_count": len(self.task_names),
            "history_policy": "no_harness_recent_n_no_environment_summary",
        }
        evidence.update(kwargs.pop("domain_evidence", {}) or {})
        return DomainTransition(
            observation=observation,
            reward=float(kwargs.pop("reward", 0.0)),
            done=bool(kwargs.pop("done", False)),
            status=kwargs.pop("status", "active"),
            phase_index=self.phase_index,
            phase_count=len(self.task_names),
            episode_success=bool(kwargs.pop("episode_success", False)),
            action_execution=kwargs.pop("action_execution", {}),
            tool_ops=kwargs.pop("tool_ops", ()),
            reward_components=kwargs.pop("reward_components", ()),
            domain_evidence=evidence,
        )


def _attest_native_splits(env, task_name: str, observed: dict[str, tuple[int, ...]]) -> None:
    manifest = _NATIVE_SPLIT_MANIFESTS[task_name]
    counts = {split: len(observed[split]) for split in SCIWORLD_SPLITS}
    if counts != manifest["split_counts"]:
        raise RuntimeError(
            f"ScienceWorld split counts changed for {task_name}: "
            f"expected {manifest['split_counts']}, observed {counts}."
        )
    flattened = [item for split in SCIWORLD_SPLITS for item in observed[split]]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError(f"ScienceWorld splits overlap for {task_name}.")
    max_variations = int(env.get_max_variations(task_name))
    if max_variations != manifest["max_variations"]:
        raise RuntimeError(
            f"ScienceWorld max variations changed for {task_name}: "
            f"expected {manifest['max_variations']}, observed {max_variations}."
        )
    structured = {split: list(observed[split]) for split in SCIWORLD_SPLITS}
    digest = hashlib.sha256(
        json.dumps(
            structured,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if digest != manifest["split_sha256"]:
        raise RuntimeError(
            f"ScienceWorld split manifest changed for {task_name}: "
            f"expected {manifest['split_sha256']}, observed {digest}."
        )


def _parse_native_action(action: str) -> tuple[str, dict[str, Any]] | None:
    text = action.strip()
    for op, regex in (("SCI_ACTION", _SCI_ACTION_RE), ("ANSWER", _ANSWER_RE)):
        match = regex.fullmatch(text)
        if match is None:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{op} payload must be valid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{op} payload must be a JSON object")
        return op, payload
    return None


def _require_payload_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_scienceworld_dependency():
    try:
        import scienceworld  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "SciWorld backend requires the optional 'scienceworld' package and "
            "Java runtime. Use AGENTMEMORY_SCIWORLD_BACKEND=fixture for static "
            "AgentMemoryGym contract tests, or install/verify SciWorld before a "
            "native smoke."
        ) from exc
    return scienceworld
