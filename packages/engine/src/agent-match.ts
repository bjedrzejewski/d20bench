import {
  DEFAULT_TACTICS,
  addBuff,
  applyDamage,
  checkBattleComplete,
  consumeResource,
  creatureDistance,
  distance,
  executeSpell,
  getActiveSpeed,
  getFootprintSize,
  getEffectiveMoveSpeed,
  getEffectiveSaveModifier,
  getHydraHeadCount,
  getAoETargets,
  hasBuff,
  hasResource,
  initBattle,
  isPositionBlocked,
  processHydraEndOfTurn,
  processTargetTurnEndOngoingEffects,
  pushLog,
  resolveAttack,
  resolveAoE,
  resolveDivineSmite,
  resolveSingleTargetSave,
  resolveSwallowAction,
  rollSaveWithBuffs,
  stabiliseDyingAlly,
  tryEscapeContainer,
  tryUseBonusActionDamageBuff,
  checkAuraEntry,
  type DamageReactionDecisionContext,
  type DamageReactionHooks,
  type DivineSmiteChoice,
  type TacticType,
  type BattleState,
} from './battlecast/engine/combat.js';
import type { Creature } from './battlecast/types/monster.js';
import { BASE_DURATIONS } from './battlecast/types/animation.js';
import { moveToDestination, moveToward } from './battlecast/engine/ai-movement.js';
import {
  beginBattlecastControlledTurn,
  executeTurn,
  processPassiveAuras,
  runOpportunityAttacks,
  type OpportunityAttackDecisionContext,
  type OpportunityAttackHooks,
} from './battlecast/engine/ai-turn.js';
import { retargetHex } from './battlecast/engine/ai-spellcasting.js';
import { canSee, estimateActionDamage, getActiveActions, getMeleeActions } from './battlecast/engine/ai-targeting.js';
import { abilityModifier, battlecastRandom, rollDice, withBattlecastRng, withBattlecastRngAsync } from './battlecast/engine/dice.js';
import { getEligibleWildShapeBeasts } from './battlecast/data/heroes.js';
import { maps } from './battlecast/data/maps.js';
import { buildMovementBlockedSet, buildSightBlockedSet } from './battlecast/types/terrain.js';
import { createBattlecastCreatures, summarizeBattlecastBattle, type BattlecastBattleSummary } from './battlecast-runner.js';
import { getAgent, type Agent, type AgentId, type Lk47Agent, type OpenRouterAgent } from './agents.js';
import {
  createBattlecastTacticAction,
  estimateAttackRollBudget,
  findLegalAction,
  generateCuttingWordsReactionActions,
  generateDamageReactionActions,
  generateLegalActions,
  generateOpportunityReactionActions,
  type LegalAction,
  type LegalActionCatalogue,
} from './legal-actions.js';
import { chooseOpenRouterAction, type OpenRouterDecisionTrace, type OpenRouterRawDecisionTrace } from './openrouter-agent.js';
import type { Lk47Simulator } from './lk-47/variants.js';
import { createRng, type RandomSeed } from './random.js';
import { hashBattlecastState, type D20benchScenario } from './scenario.js';
import type { ReplayEvent, ReplayEventController } from './replay.js';

export interface AgentMatchSpec {
  scenario: D20benchScenario;
  seed: RandomSeed;
  redAgent: AgentId;
  blueAgent: AgentId;
  maxRounds?: number;
  llmActionSpace?: LlmActionSpace;
  llmDecisionTraceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
}

export type LlmActionSpace = 'primitive' | 'battlecast-full-turn' | 'actual-actions-v1';

export interface AgentMatchResult {
  matchId: string;
  scenarioId: string;
  seed: RandomSeed;
  redAgent: AgentId;
  blueAgent: AgentId;
  winner: BattleState['winner'];
  redScore: number;
  blueScore: number;
  finalStateHash: string;
  state: BattleState;
  summary: BattlecastBattleSummary;
  replay: ReplayEvent[];
}

export function runAgentMatch(spec: AgentMatchSpec): AgentMatchResult {
  const maxRounds = spec.maxRounds ?? 100;
  const matchId = createMatchId(spec);
  const red = getAgent(spec.redAgent);
  const blue = getAgent(spec.blueAgent);
  const agentRng = createRng(`${spec.seed}:agents:${spec.redAgent}:${spec.blueAgent}`);

  return withBattlecastRng(spec.seed, () => {
    const state = initAgentBattleState(spec.scenario);
    state.matchMaxRounds = maxRounds;
    configureBattlecastTacticsForAgents(state, red, blue);
    const replay: ReplayEvent[] = [];
    replay.push({
      type: 'match_started',
      matchId,
      scenario: {
        id: spec.scenario.id,
        rulesetId: spec.scenario.rulesetId,
        dataPackId: spec.scenario.dataPackId,
        scenarioVersion: spec.scenario.scenarioVersion,
      },
      seed: spec.seed,
      redAgent: spec.redAgent,
      blueAgent: spec.blueAgent,
      stateHash: hashBattlecastState(state),
    });

    while (!state.isComplete && state.round <= maxRounds) {
      replay.push({ type: 'round_started', matchId, round: state.round, stateHash: hashBattlecastState(state) });
      for (let i = 0; i < state.initiativeOrder.length; i += 1) {
        if (state.isComplete) break;
        state.turnIndex = i;
        const active = state.creatures.find((creature) => creature.id === state.initiativeOrder[i]);
        if (!active || !active.isAlive) continue;

        const agent = active.team === 'red' ? red : blue;
        const turnStart = agent.kind === 'battlecast-tactic'
          ? { canAct: true, logsBefore: state.logs.length, eventsBefore: state.events.length, turnStartAlreadyProcessed: false }
          : processManualAgentTurnStart({
              state,
              active,
              agent,
              replay,
              matchId,
              autoClassFeatures: !usesActualActions(agent, spec.llmActionSpace),
              actionSpace: actionSpaceForAgent(agent, spec.llmActionSpace ?? 'primitive'),
            });
        if (!turnStart.canAct) continue;

        const catalogue = createActionCatalogueForAgent(state, active, agent, spec.llmActionSpace ?? 'primitive');
        replay.push({
          type: 'turn_started',
          matchId,
          round: state.round,
          turnIndex: state.turnIndex,
          activeCreatureId: active.id,
          activeCreatureName: active.displayName,
          controller: describeAgentController(agent),
          actionSpace: catalogue.actionSpace,
          legalActions: catalogue.actions,
          actionEconomy: catalogue.actionEconomy,
          stateHash: hashBattlecastState(state),
        });

        const logsBefore = turnStart.logsBefore;
        const eventsBefore = turnStart.eventsBefore;
        const { requestedActionId, acceptedAction, llmTrace } = applyAgentTurn({
          state,
          active,
          agent,
          catalogue,
          agentRng,
          turnStartAlreadyProcessed: turnStart.turnStartAlreadyProcessed,
        });
        checkBattleComplete(state);

        replay.push({
          type: 'action_resolved',
          matchId,
          round: state.round,
          turnIndex: state.turnIndex,
          activeCreatureId: active.id,
          agentId: agent.id,
          requestedActionId,
          acceptedAction,
          llmTrace,
          logs: state.logs.slice(logsBefore),
          events: state.events.slice(eventsBefore),
          stateHash: hashBattlecastState(state),
        });
      }

      replay.push({ type: 'round_ended', matchId, round: state.round, stateHash: hashBattlecastState(state) });
      state.round += 1;
    }

    if (!state.isComplete) {
      finishByRemainingHp(state);
    }

    const finalStateHash = hashBattlecastState(state);
    replay.push({
      type: 'match_finished',
      matchId,
      winner: state.winner,
      finalStateHash,
      rounds: state.round,
    });

    return {
      matchId,
      scenarioId: spec.scenario.id,
      seed: spec.seed,
      redAgent: spec.redAgent,
      blueAgent: spec.blueAgent,
      winner: state.winner,
      redScore: scoreForWinner(state.winner, 'red'),
      blueScore: scoreForWinner(state.winner, 'blue'),
      finalStateHash,
      state,
      summary: summarizeBattlecastBattle(state),
      replay,
    };
  });
}

export async function runAgentMatchAsync(spec: AgentMatchSpec): Promise<AgentMatchResult> {
  const maxRounds = spec.maxRounds ?? 100;
  const matchId = createMatchId(spec);
  const red = getAgent(spec.redAgent);
  const blue = getAgent(spec.blueAgent);
  if ((red.kind === 'lk-47' || blue.kind === 'lk-47') && spec.llmActionSpace !== 'actual-actions-v1') {
    throw new Error('lk-47 agents play only with llmActionSpace actual-actions-v1');
  }
  const agentRng = createRng(`${spec.seed}:agents:${spec.redAgent}:${spec.blueAgent}`);
  const battleRng = createRng(spec.seed);

  return withBattlecastRngAsync(battleRng, async () => {
    const state = initAgentBattleState(spec.scenario);
    state.matchMaxRounds = maxRounds;
    configureBattlecastTacticsForAgents(state, red, blue);
    const replay: ReplayEvent[] = [];
    replay.push({
      type: 'match_started',
      matchId,
      scenario: {
        id: spec.scenario.id,
        rulesetId: spec.scenario.rulesetId,
        dataPackId: spec.scenario.dataPackId,
        scenarioVersion: spec.scenario.scenarioVersion,
      },
      seed: spec.seed,
      redAgent: spec.redAgent,
      blueAgent: spec.blueAgent,
      stateHash: hashBattlecastState(state),
    });

    while (!state.isComplete && state.round <= maxRounds) {
      replay.push({ type: 'round_started', matchId, round: state.round, stateHash: hashBattlecastState(state) });
      for (let i = 0; i < state.initiativeOrder.length; i += 1) {
        if (state.isComplete) break;
        state.turnIndex = i;
        const active = state.creatures.find((creature) => creature.id === state.initiativeOrder[i]);
        if (!active || !active.isAlive) continue;

        const agent = active.team === 'red' ? red : blue;
        const turnStart = agent.kind === 'battlecast-tactic'
          ? { canAct: true, logsBefore: state.logs.length, eventsBefore: state.events.length, turnStartAlreadyProcessed: false }
          : processManualAgentTurnStart({
              state,
              active,
              agent,
              replay,
              matchId,
              autoClassFeatures: !usesActualActions(agent, spec.llmActionSpace),
              actionSpace: actionSpaceForAgent(agent, spec.llmActionSpace ?? 'primitive'),
            });
        if (!turnStart.canAct) continue;

        if (isStepwiseAgent(agent) && usesActualActions(agent, spec.llmActionSpace)) {
          await runStepwiseTurn({
            state,
            active,
            agent,
            controllers: { red, blue },
            agentRng,
            battleRng,
            turnStart,
            replay,
            matchId,
            traceSink: spec.llmDecisionTraceSink,
          });
          checkBattleComplete(state);
          continue;
        }

        if (agent.kind === 'battlecast-tactic' && (spec.llmActionSpace ?? 'primitive') === 'actual-actions-v1') {
          const catalogue = createActionCatalogueForAgent(state, active, agent, spec.llmActionSpace ?? 'primitive');
          replay.push({
            type: 'turn_started',
            matchId,
            round: state.round,
            turnIndex: state.turnIndex,
            activeCreatureId: active.id,
            activeCreatureName: active.displayName,
            controller: describeAgentController(agent),
            actionSpace: catalogue.actionSpace,
            legalActions: catalogue.actions,
            actionEconomy: catalogue.actionEconomy,
            stateHash: hashBattlecastState(state),
          });

          const logsBefore = turnStart.logsBefore;
          const eventsBefore = turnStart.eventsBefore;
          // Opportunity reactions are requested at trigger time through the
          // snapshot/retry loop instead of predeclaring before every turn, so
          // models are only prompted when an opportunity attack actually fires.
          const reactions: OpportunityReactionMap = new Map();
          const opportunityAttacks = createOpportunityAttackHooks({
            state,
            replay,
            matchId,
            controllers: { red, blue },
            decisions: reactions,
            missingDecision: 'request',
          });
          const damageReactionDecisions: DamageReactionMap = new Map();
          await applyBattlecastTacticTurnWithDynamicDamageReactions({
            state,
            active,
            tactic: agent.tactic,
            turnStartAlreadyProcessed: turnStart.turnStartAlreadyProcessed,
            opportunityAttacks,
            opportunityDecisions: reactions,
            replay,
            matchId,
            controllers: { red, blue },
            decisions: damageReactionDecisions,
            agentRng,
            battleRng,
            traceSink: spec.llmDecisionTraceSink,
          });
          checkBattleComplete(state);

          const acceptedAction = createBattlecastTacticAction(agent.tactic);
          replay.push({
            type: 'action_resolved',
            matchId,
            round: state.round,
            turnIndex: state.turnIndex,
            activeCreatureId: active.id,
            agentId: agent.id,
            requestedActionId: acceptedAction.id,
            acceptedAction,
            logs: state.logs.slice(logsBefore),
            events: state.events.slice(eventsBefore),
            stateHash: hashBattlecastState(state),
          });
          continue;
        }

        const catalogue = createActionCatalogueForAgent(state, active, agent, spec.llmActionSpace ?? 'primitive');
        replay.push({
          type: 'turn_started',
          matchId,
          round: state.round,
          turnIndex: state.turnIndex,
          activeCreatureId: active.id,
          activeCreatureName: active.displayName,
          controller: describeAgentController(agent),
          actionSpace: catalogue.actionSpace,
          legalActions: catalogue.actions,
          actionEconomy: catalogue.actionEconomy,
          stateHash: hashBattlecastState(state),
        });

        const logsBefore = turnStart.logsBefore;
        const eventsBefore = turnStart.eventsBefore;
        const { requestedActionId, acceptedAction, llmTrace } = await applyAgentTurnAsync({
          state,
          active,
          agent,
          catalogue,
          agentRng,
          turnStartAlreadyProcessed: turnStart.turnStartAlreadyProcessed,
          matchId,
          round: state.round,
          turnIndex: state.turnIndex,
          traceSink: spec.llmDecisionTraceSink,
        });
        checkBattleComplete(state);

        replay.push({
          type: 'action_resolved',
          matchId,
          round: state.round,
          turnIndex: state.turnIndex,
          activeCreatureId: active.id,
          agentId: agent.id,
          requestedActionId,
          acceptedAction,
          llmTrace,
          logs: state.logs.slice(logsBefore),
          events: state.events.slice(eventsBefore),
          stateHash: hashBattlecastState(state),
        });
      }

      replay.push({ type: 'round_ended', matchId, round: state.round, stateHash: hashBattlecastState(state) });
      state.round += 1;
    }

    if (!state.isComplete) {
      finishByRemainingHp(state);
    }

    const finalStateHash = hashBattlecastState(state);
    replay.push({
      type: 'match_finished',
      matchId,
      winner: state.winner,
      finalStateHash,
      rounds: state.round,
    });

    return {
      matchId,
      scenarioId: spec.scenario.id,
      seed: spec.seed,
      redAgent: spec.redAgent,
      blueAgent: spec.blueAgent,
      winner: state.winner,
      redScore: scoreForWinner(state.winner, 'red'),
      blueScore: scoreForWinner(state.winner, 'blue'),
      finalStateHash,
      state,
      summary: summarizeBattlecastBattle(state),
      replay,
    };
  });
}

function initAgentBattleState(scenario: D20benchScenario): BattleState {
  const state = initBattle(createBattlecastCreatures(scenario.combatants, true), scenario.gridSize);
  const map = scenario.mapId ? maps.find((candidate) => candidate.id === scenario.mapId) : undefined;
  if (scenario.mapId && !map) {
    throw new Error(`unknown Battlecast map id: ${scenario.mapId}`);
  }
  state.teamTactics = scenario.teamTactics ?? DEFAULT_TACTICS;
  state.terrainBlocked = buildMovementBlockedSet(map?.terrain);
  state.terrainSightBlocked = buildSightBlockedSet(map?.terrain);
  scenario.setupBattleState?.(state);
  return state;
}

function configureBattlecastTacticsForAgents(state: BattleState, red: Agent, blue: Agent): void {
  state.teamTactics = {
    ...state.teamTactics,
    red: red.kind === 'battlecast-tactic' ? red.tactic : state.teamTactics.red,
    blue: blue.kind === 'battlecast-tactic' ? blue.tactic : state.teamTactics.blue,
  };
}

function describeAgentController(agent: Agent): ReplayEventController {
  if (agent.kind === 'battlecast-tactic') {
    return {
      mode: 'battlecast-tactic',
      agentId: agent.id,
      tactic: agent.tactic,
    };
  }

  if (agent.kind === 'openrouter-llm') {
    return {
      mode: 'openrouter-llm',
      agentId: agent.id,
      model: agent.model,
    };
  }

  if (agent.kind === 'lk-47') {
    return {
      mode: 'lk-47',
      agentId: agent.id,
    };
  }

  return {
    mode: 'legal-action',
    agentId: agent.id,
  };
}

function actionSpaceForAgent(agent: Agent, llmActionSpace: LlmActionSpace): LlmActionSpace {
  if (agent.kind === 'lk-47') return 'actual-actions-v1';
  return agent.kind === 'openrouter-llm' ? llmActionSpace : 'primitive';
}

type StepwiseAgent = OpenRouterAgent | Lk47Agent;

interface StepwiseSelection {
  requestedActionId: string;
  acceptedAction: LegalAction;
  trace?: OpenRouterDecisionTrace;
}

function isStepwiseAgent(agent: Agent): agent is StepwiseAgent {
  return agent.kind === 'openrouter-llm' || agent.kind === 'lk-47';
}

function usesActualActions(agent: Agent, llmActionSpace: LlmActionSpace | undefined): boolean {
  return agent.kind === 'lk-47' || (agent.kind === 'openrouter-llm' && llmActionSpace === 'actual-actions-v1');
}

// LK-47 receives the state, the active creature and the legal menu, and must pick from the menu, like an LLM.
async function chooseStepwiseAction(
  agent: StepwiseAgent,
  context: Parameters<typeof chooseOpenRouterAction>[1] & { actualTurn?: ActualTurnContext },
): Promise<StepwiseSelection> {
  const { actualTurn, ...llmContext } = context;
  if (agent.kind === 'openrouter-llm') return chooseOpenRouterAction(agent, llmContext);
  const requested = await agent.policy.chooseAction({
    state: context.state,
    activeCreature: context.activeCreature,
    catalogue: context.catalogue,
    turn: actualTurn && structuredClone(actualTurn),
    simulator: lk47Simulator,
  });
  const acceptedAction = findLegalAction(context.catalogue, requested.id);
  if (!acceptedAction) {
    throw new Error(`${agent.id} chose ${requested.id}, which is not in the legal menu`);
  }
  return { requestedActionId: requested.id, acceptedAction };
}

const simulationControllers: MatchControllers = { red: getAgent('battlecast.smart'), blue: getAgent('battlecast.smart') };

// What LK-47 may do to its private copies of a battle: the rule paths a match uses, with no replay,
// no prompts, and the scripted bots' default reactions on both sides.
export const lk47Simulator: Lk47Simulator = {
  actionsInPlace: (state, active, turn) =>
    generateLegalActions(state, { ...active, movementRemaining: 0 }, { includeActualActions: true, actualTurnContext: turn }),
  async applyAction(state, active, action, turn, rng) {
    const result = await applyActualLegalAction({
      state,
      active,
      action,
      agent: simulationControllers.red,
      actualTurn: turn,
      controllers: simulationControllers,
      agentRng: rng,
      battleRng: rng,
      replay: [],
      matchId: 'lk-47-simulation',
      turnStep: 0,
    });
    checkBattleComplete(state);
    return result.ended;
  },
  finishTurn(state, active) {
    finishManualTurnEnd(state, active);
    checkBattleComplete(state);
  },
  playScriptedTurn(state, active, tactic) {
    applyBattlecastTacticTurn(state, active, tactic);
    checkBattleComplete(state);
  },
  beginTurn(state, active) {
    const start = processManualAgentTurnStart({
      state,
      active,
      agent: simulationControllers.red,
      replay: [],
      matchId: 'lk-47-simulation',
      autoClassFeatures: false,
      actionSpace: 'actual-actions-v1',
    });
    return start.canAct ? newActualTurn(active) : undefined;
  },
};

interface AgentTurnInput {
  state: BattleState;
  active: Creature;
  agent: Agent;
  catalogue: LegalActionCatalogue;
  agentRng: ReturnType<typeof createRng>;
  turnStartAlreadyProcessed: boolean;
  matchId?: string;
  round?: number;
  turnIndex?: number;
  turnStep?: number;
  traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
}

interface AgentTurnResult {
  requestedActionId: string;
  acceptedAction: LegalAction;
  llmTrace?: OpenRouterDecisionTrace;
}

interface ManualTurnStartResult {
  canAct: boolean;
  logsBefore: number;
  eventsBefore: number;
  turnStartAlreadyProcessed: boolean;
}

export interface ActualTurnContext {
  attackRollsRemaining: number;
  attackActionStarted: boolean;
  flurryStrikesRemaining: number;
  pendingSmite?: {
    targetId: string;
    targetName: string;
    actionName: string;
    isCritical: boolean;
  };
  disengaged: boolean;
  ended: boolean;
}

interface ActualActionApplyResult {
  ended: boolean;
}

interface MatchControllers {
  red: Agent;
  blue: Agent;
}

interface PreparedOpportunityReaction {
  agent: StepwiseAgent;
  catalogue: LegalActionCatalogue;
  requestedActionId: string;
  acceptedAction: Extract<LegalAction, { type: 'reaction' }>;
  llmTrace?: OpenRouterDecisionTrace;
}

type OpportunityReactionMap = Map<string, PreparedOpportunityReaction>;

interface PreparedDamageReaction {
  agent: StepwiseAgent;
  catalogue: LegalActionCatalogue;
  requestedActionId: string;
  acceptedAction: Extract<LegalAction, { type: 'reaction' }>;
  llmTrace?: OpenRouterDecisionTrace;
}

type DamageReactionMap = Map<string, PreparedDamageReaction>;

interface PendingDamageReactionRequest {
  triggerKey: string;
  context: DamageReactionDecisionContext;
  agent: StepwiseAgent;
}

class PendingDamageReactionDecision extends Error {
  constructor(readonly request: PendingDamageReactionRequest) {
    super('OpenRouter damage reaction decision required');
    this.name = 'PendingDamageReactionDecision';
  }
}

interface PendingOpportunityReactionRequest {
  reactorId: string;
  moverId: string;
  triggerCell: { x: number; y: number };
}

class PendingOpportunityReactionDecision extends Error {
  constructor(readonly request: PendingOpportunityReactionRequest) {
    super('OpenRouter opportunity reaction decision required');
    this.name = 'PendingOpportunityReactionDecision';
  }
}

function processManualAgentTurnStart(input: {
  state: BattleState;
  active: Creature;
  agent: Agent;
  replay: ReplayEvent[];
  matchId: string;
  autoClassFeatures?: boolean;
  actionSpace?: LlmActionSpace;
}): ManualTurnStartResult {
  const logsBefore = input.state.logs.length;
  const eventsBefore = input.state.events.length;
  const canAct = beginBattlecastControlledTurn(input.state, input.active, {
    autoClassFeatures: input.autoClassFeatures,
  });
  checkBattleComplete(input.state);

  if (canAct && !input.state.isComplete) {
    processPassiveAuras(input.state, input.active);
    checkBattleComplete(input.state);
  }

  if (canAct && input.active.isAlive && !input.state.isComplete) {
    return {
      canAct: true,
      logsBefore,
      eventsBefore,
      turnStartAlreadyProcessed: true,
    };
  }

  const acceptedAction: LegalAction = { id: 'end_turn', type: 'end_turn' };
  input.replay.push({
    type: 'turn_started',
    matchId: input.matchId,
    round: input.state.round,
    turnIndex: input.state.turnIndex,
    activeCreatureId: input.active.id,
    activeCreatureName: input.active.displayName,
    controller: describeAgentController(input.agent),
    actionSpace: input.actionSpace,
    legalActions: [acceptedAction],
    stateHash: hashBattlecastState(input.state),
  });
  input.replay.push({
    type: 'action_resolved',
    matchId: input.matchId,
    round: input.state.round,
    turnIndex: input.state.turnIndex,
    activeCreatureId: input.active.id,
    agentId: input.agent.id,
    requestedActionId: 'automatic:turn_start',
    acceptedAction,
    logs: input.state.logs.slice(logsBefore),
    events: input.state.events.slice(eventsBefore),
    stateHash: hashBattlecastState(input.state),
  });

  return {
    canAct: false,
    logsBefore,
    eventsBefore,
    turnStartAlreadyProcessed: true,
  };
}

function applyAgentTurn(input: AgentTurnInput): AgentTurnResult {
  if (input.agent.kind === 'battlecast-tactic') {
    const acceptedAction = createBattlecastTacticAction(input.agent.tactic);
    applyBattlecastTacticTurn(input.state, input.active, input.agent.tactic);
    return { requestedActionId: acceptedAction.id, acceptedAction };
  }

  if (input.agent.kind === 'openrouter-llm' || input.agent.kind === 'lk-47') {
    throw new Error(`${input.agent.id} requires runAgentMatchAsync`);
  }

  const requested = input.agent.chooseAction({
    state: input.state,
    activeCreature: input.active,
    catalogue: input.catalogue,
    rng: input.agentRng,
  });
  const acceptedAction = findLegalAction(input.catalogue, requested.id) ?? ({ id: 'end_turn', type: 'end_turn' } satisfies LegalAction);
  applyLegalAction(input.state, input.active, acceptedAction, input.agent, input.turnStartAlreadyProcessed);
  return { requestedActionId: requested.id, acceptedAction };
}

async function applyAgentTurnAsync(input: AgentTurnInput): Promise<AgentTurnResult> {
  if (input.agent.kind === 'openrouter-llm') {
    const selection = await chooseOpenRouterAction(input.agent, {
      state: input.state,
      activeCreature: input.active,
      catalogue: input.catalogue,
      rng: input.agentRng,
      traceMeta: input.matchId && input.round !== undefined && input.turnIndex !== undefined ? {
        matchId: input.matchId,
        round: input.round,
        turnIndex: input.turnIndex,
        turnStep: input.turnStep,
        activeCreatureId: input.active.id,
        activeCreatureName: input.active.displayName,
        agentId: input.agent.id,
      } : undefined,
      traceSink: input.traceSink,
    });
    applyLegalAction(input.state, input.active, selection.acceptedAction, input.agent, input.turnStartAlreadyProcessed);
    return {
      requestedActionId: selection.requestedActionId,
      acceptedAction: selection.acceptedAction,
      llmTrace: selection.trace,
    };
  }

  return applyAgentTurn(input);
}

function newActualTurn(active: Creature): ActualTurnContext {
  return {
    attackRollsRemaining: estimateAttackRollBudget(active),
    attackActionStarted: false,
    flurryStrikesRemaining: 0,
    disengaged: false,
    ended: false,
  };
}

async function runStepwiseTurn(input: {
  state: BattleState;
  active: Creature;
  agent: StepwiseAgent;
  controllers: MatchControllers;
  agentRng: ReturnType<typeof createRng>;
  battleRng: ReturnType<typeof createRng>;
  turnStart: ManualTurnStartResult;
  replay: ReplayEvent[];
  matchId: string;
  traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
}): Promise<void> {
  const actualTurn = newActualTurn(input.active);
  const maxSteps = 12;
  let step = 0;

  while (!input.state.isComplete && !actualTurn.ended && step < maxSteps) {
    const active = input.state.creatures.find((creature) => creature.id === input.active.id);
    if (!active || !active.isAlive) break;
    const catalogue = createActionCatalogueForAgent(
      input.state,
      active,
      input.agent,
      'actual-actions-v1',
      actualTurn,
    );
    const nonEndActions = catalogue.actions.filter((action) => action.type !== 'end_turn');
    input.replay.push({
      type: 'turn_started',
      matchId: input.matchId,
      round: input.state.round,
      turnIndex: input.state.turnIndex,
      turnStep: step,
      activeCreatureId: active.id,
      activeCreatureName: active.displayName,
      controller: describeAgentController(input.agent),
      actionSpace: catalogue.actionSpace,
      legalActions: catalogue.actions,
      actionEconomy: catalogue.actionEconomy,
      stateHash: hashBattlecastState(input.state),
    });

    if (nonEndActions.length === 0) {
      const acceptedAction: LegalAction = { id: 'end_turn', type: 'end_turn' };
      const logsBefore = step === 0 ? input.turnStart.logsBefore : input.state.logs.length;
      const eventsBefore = step === 0 ? input.turnStart.eventsBefore : input.state.events.length;
      await applyActualLegalAction({
        state: input.state,
        active,
        action: acceptedAction,
        agent: input.agent,
        actualTurn,
        controllers: input.controllers,
        agentRng: input.agentRng,
        battleRng: input.battleRng,
        replay: input.replay,
        matchId: input.matchId,
        turnStep: step,
        traceSink: input.traceSink,
      });
      input.replay.push({
        type: 'action_resolved',
        matchId: input.matchId,
        round: input.state.round,
        turnIndex: input.state.turnIndex,
        turnStep: step,
        activeCreatureId: active.id,
        agentId: input.agent.id,
        requestedActionId: 'automatic:no_legal_actions',
        acceptedAction,
        logs: input.state.logs.slice(logsBefore),
        events: input.state.events.slice(eventsBefore),
        stateHash: hashBattlecastState(input.state),
      });
      actualTurn.ended = true;
      break;
    }

    const logsBefore = step === 0 ? input.turnStart.logsBefore : input.state.logs.length;
    const eventsBefore = step === 0 ? input.turnStart.eventsBefore : input.state.events.length;
    let selection: StepwiseSelection;
    try {
      selection = await chooseStepwiseAction(input.agent, {
        state: input.state,
        activeCreature: active,
        catalogue,
        actualTurn,
        rng: input.agentRng,
        traceMeta: {
          matchId: input.matchId,
          round: input.state.round,
          turnIndex: input.state.turnIndex,
          turnStep: step,
          activeCreatureId: active.id,
          activeCreatureName: active.displayName,
          agentId: input.agent.id,
        },
        traceSink: input.traceSink,
      });
    } catch (error) {
      // Repeated malformed output costs this creature its turn instead of
      // aborting the whole match. Provider/network failures still throw so
      // an outage stays loud rather than silently skewing results.
      if (!(error instanceof Error) || !error.message.includes('did not produce a legal action')) {
        throw error;
      }
      pushLog(input.state, {
        round: input.state.round,
        turn: input.state.turnIndex,
        actor: active.displayName,
        action: 'End Turn',
        details: `${active.displayName} ends their turn because the controlling model failed to choose a legal action.`,
        type: 'info',
      });
      const acceptedAction: LegalAction = { id: 'end_turn', type: 'end_turn' };
      input.replay.push({
        type: 'action_resolved',
        matchId: input.matchId,
        round: input.state.round,
        turnIndex: input.state.turnIndex,
        turnStep: step,
        activeCreatureId: active.id,
        agentId: input.agent.id,
        requestedActionId: 'automatic:decision_failure',
        acceptedAction,
        logs: input.state.logs.slice(logsBefore),
        events: input.state.events.slice(eventsBefore),
        stateHash: hashBattlecastState(input.state),
      });
      actualTurn.ended = true;
      break;
    }
    const applyResult = await applyActualLegalAction({
      state: input.state,
      active,
      action: selection.acceptedAction,
      agent: input.agent,
      actualTurn,
      controllers: input.controllers,
      agentRng: input.agentRng,
      battleRng: input.battleRng,
      replay: input.replay,
      matchId: input.matchId,
      turnStep: step,
      traceSink: input.traceSink,
    });
    checkBattleComplete(input.state);
    input.replay.push({
      type: 'action_resolved',
      matchId: input.matchId,
      round: input.state.round,
      turnIndex: input.state.turnIndex,
      turnStep: step,
      activeCreatureId: active.id,
      agentId: input.agent.id,
      requestedActionId: selection.requestedActionId,
      acceptedAction: selection.acceptedAction,
      llmTrace: selection.trace,
      logs: input.state.logs.slice(logsBefore),
      events: input.state.events.slice(eventsBefore),
      stateHash: hashBattlecastState(input.state),
    });
    actualTurn.ended = applyResult.ended;
    step += 1;
  }

  if (!actualTurn.ended && !input.state.isComplete && input.active.isAlive) {
    pushLog(input.state, {
      round: input.state.round,
      turn: input.state.turnIndex,
      actor: input.active.displayName,
      action: 'End Turn',
      details: `${input.active.displayName} ends their turn after reaching the step limit.`,
      type: 'info',
    });
  }
  const active = input.state.creatures.find((creature) => creature.id === input.active.id) ?? input.active;
  finishManualTurnEnd(input.state, active);
  checkBattleComplete(input.state);
}

function applyBattlecastTacticTurn(
  state: BattleState,
  active: Creature,
  tactic: TacticType,
  turnStartAlreadyProcessed = false,
  opportunityAttacks?: OpportunityAttackHooks,
  damageReactions?: DamageReactionHooks,
): void {
  state.teamTactics = {
    ...state.teamTactics,
    [active.team]: tactic,
  };
  active.stats.roundsSurvived = state.round;
  executeTurn(state, active, { turnStartAlreadyProcessed, opportunityAttacks, damageReactions });
}

async function applyBattlecastTacticTurnWithDynamicDamageReactions(input: {
  state: BattleState;
  active: Creature;
  tactic: TacticType;
  turnStartAlreadyProcessed: boolean;
  opportunityAttacks: OpportunityAttackHooks;
  opportunityDecisions?: OpportunityReactionMap;
  replay: ReplayEvent[];
  matchId: string;
  controllers: MatchControllers;
  decisions: DamageReactionMap;
  agentRng: ReturnType<typeof createRng>;
  battleRng: ReturnType<typeof createRng>;
  traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
}): Promise<void> {
  const maxReactionPrompts = 20;
  for (let attempt = 0; attempt <= maxReactionPrompts; attempt += 1) {
    const stateSnapshot = cloneBattleState(input.state);
    const replayLength = input.replay.length;
    const rngSnapshot = input.battleRng.snapshot();
    const damageReactions = createDamageReactionHooks({
      state: input.state,
      replay: input.replay,
      matchId: input.matchId,
      controllers: input.controllers,
      decisions: input.decisions,
    });
    const active = input.state.creatures.find((creature) => creature.id === input.active.id);
    if (!active) {
      throw new Error(`Active creature ${input.active.id} disappeared during ${input.matchId}`);
    }

    try {
      applyBattlecastTacticTurn(
        input.state,
        active,
        input.tactic,
        input.turnStartAlreadyProcessed,
        input.opportunityAttacks,
        damageReactions,
      );
      return;
    } catch (error) {
      if (error instanceof PendingOpportunityReactionDecision && input.opportunityDecisions) {
        const prepared = await chooseOpportunityReactionAtTrigger({
          state: input.state,
          request: error.request,
          controllers: input.controllers,
          agentRng: input.agentRng,
          matchId: input.matchId,
          traceSink: input.traceSink,
        });
        input.opportunityDecisions.set(prepared.key, prepared.decision);
        restoreBattleState(input.state, stateSnapshot);
        input.replay.length = replayLength;
        input.battleRng.restore(rngSnapshot);
        continue;
      }
      if (!(error instanceof PendingDamageReactionDecision)) throw error;
      const prepared = await chooseDamageReactionAtTrigger({
        state: input.state,
        request: error.request,
        agentRng: input.agentRng,
        matchId: input.matchId,
        traceSink: input.traceSink,
      });
      input.decisions.set(error.request.triggerKey, prepared);
      restoreBattleState(input.state, stateSnapshot);
      input.replay.length = replayLength;
      input.battleRng.restore(rngSnapshot);
    }
  }

  throw new Error(`Exceeded ${maxReactionPrompts} pending damage reaction prompts in ${input.matchId}`);
}

async function chooseOpportunityReactionAtTrigger(input: {
  state: BattleState;
  request: PendingOpportunityReactionRequest;
  controllers: MatchControllers;
  agentRng: ReturnType<typeof createRng>;
  matchId: string;
  traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
}): Promise<{ key: string; decision: PreparedOpportunityReaction }> {
  const reactor = input.state.creatures.find((creature) => creature.id === input.request.reactorId);
  const mover = input.state.creatures.find((creature) => creature.id === input.request.moverId);
  if (!reactor || !mover) {
    throw new Error(`Opportunity reaction trigger references missing creatures in ${input.matchId}`);
  }
  const agent = controllerForCreature(reactor, input.controllers);
  if (!isStepwiseAgent(agent)) {
    throw new Error(`Opportunity reaction trigger for non-LLM reactor ${reactor.id} in ${input.matchId}`);
  }
  const decision = await chooseOpportunityReaction({
    state: input.state,
    reactor,
    mover,
    triggerCell: input.request.triggerCell,
    agent,
    agentRng: input.agentRng,
    matchId: input.matchId,
    traceSink: input.traceSink,
  });
  return {
    key: opportunityReactionKey(reactor, mover),
    decision: decision ?? declinedOpportunityReaction(agent, reactor, mover, input.request.triggerCell),
  };
}

function declinedOpportunityReaction(
  agent: StepwiseAgent,
  reactor: Creature,
  mover: Creature,
  triggerCell: { x: number; y: number },
): PreparedOpportunityReaction {
  const originalPosition = { ...mover.position };
  mover.position = { ...triggerCell };
  try {
    const catalogue = generateOpportunityReactionActions(reactor, mover);
    const decline = catalogue.actions.find((action) =>
      action.type === 'reaction' && action.reaction === 'decline'
    );
    if (!decline || decline.type !== 'reaction') {
      throw new Error(`No decline reaction available for ${reactor.id} against ${mover.id}`);
    }
    return {
      agent,
      catalogue,
      requestedActionId: decline.id,
      acceptedAction: decline,
    };
  } finally {
    mover.position = originalPosition;
  }
}

function cloneBattleState(state: BattleState): BattleState {
  return structuredClone(state) as BattleState;
}

function restoreBattleState(target: BattleState, snapshot: BattleState): void {
  for (const key of Object.keys(target) as Array<keyof BattleState>) {
    delete target[key];
  }
  Object.assign(target, cloneBattleState(snapshot));
}

function createActionCatalogueForAgent(
  state: BattleState,
  active: Creature,
  agent: Agent,
  llmActionSpace: LlmActionSpace,
  actualTurnContext?: ActualTurnContext,
): LegalActionCatalogue {
  return agent.kind === 'battlecast-tactic'
    ? {
        activeCreatureId: active.id,
        activeCreatureName: active.displayName,
        actionSpace: 'battlecast-full-turn',
        actions: [createBattlecastTacticAction(agent.tactic)],
      }
    : generateLegalActions(state, active, {
        includeBattlecastFullTurnActions: agent.kind === 'openrouter-llm' && llmActionSpace === 'battlecast-full-turn',
        includeActualActions: usesActualActions(agent, llmActionSpace),
        actualTurnContext,
      });
}

async function applyActualLegalAction(input: {
  state: BattleState;
  active: Creature;
  action: LegalAction;
  agent: Agent;
  actualTurn: ActualTurnContext;
  controllers: MatchControllers;
  agentRng: ReturnType<typeof createRng>;
  battleRng: ReturnType<typeof createRng>;
  replay: ReplayEvent[];
  matchId: string;
  turnStep: number;
  traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
}): Promise<ActualActionApplyResult> {
  const {
    state,
    active,
    action,
    agent,
    actualTurn,
  } = input;
  if (action.type === 'end_turn') {
    applyLegalAction(state, active, action, agent, true);
    actualTurn.ended = true;
    return { ended: true };
  }

  if (action.type === 'move_toward') {
    const before = { ...active.position };
    applyLegalAction(state, active, action, agent, true);
    await processPostMoveEffects(input, before);
    return { ended: false };
  }

  if (action.type === 'move_to') {
    const before = { ...active.position };
    moveToDestination(active, action.destination, state);
    if (before.x !== active.position.x || before.y !== active.position.y) {
      active.hasMovedThisTurn = true;
      pushLog(state, {
        round: state.round,
        turn: state.turnIndex,
        actor: active.displayName,
        action: 'Move',
        details: `${active.displayName} moves to (${active.position.x},${active.position.y}).`,
        type: 'move',
      });
    }
    await processPostMoveEffects(input, before);
    return { ended: false };
  }

  if (action.type === 'dash') {
    applyDashAction(state, active, action, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'dodge') {
    applyDodgeAction(state, active, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'help') {
    applyHelpAction(state, active, action, agent, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'stabilise') {
    applyStabiliseAction(state, active, action, agent, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'escape_container') {
    applyEscapeContainerAction(state, active, action, agent, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'linked_bonus_damage') {
    applyLinkedBonusDamageAction(state, active, action, agent);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'spell_retarget') {
    applySpellRetargetAction(state, active, action, agent);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'disengage') {
    applyDisengageAction(state, active, action, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'class_feature') {
    applyClassFeatureAction(state, active, action, agent, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'smite') {
    applySmiteAction(state, active, action, agent, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'attack') {
    applyAttackAction(state, active, action, agent, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'random_ray') {
    applyRandomRayAction(state, active, action, agent, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  if (action.type === 'spell') {
    applySpellAction(state, active, action, agent, actualTurn);
    return { ended: shouldEndActualTurn(active, actualTurn) };
  }

  applyLegalAction(state, active, action, agent, true);
  actualTurn.ended = true;
  return { ended: true };
}

function applyAttackAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'attack' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  const target = state.creatures.find((creature) => creature.id === action.targetId);
  const battlecastAction = getActiveActions(active).find((candidate) => candidate.name === action.actionName);
  if (!target || !battlecastAction) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  const result = resolveAttack(state, active, target, battlecastAction, { deferSmite: true });
  if (result?.smiteEligible && battlecastAction.smiteOnHit) {
    actualTurn.pendingSmite = {
      targetId: target.id,
      targetName: target.displayName,
      actionName: battlecastAction.name,
      isCritical: result.critical,
    };
  } else {
    actualTurn.pendingSmite = undefined;
  }
  actualTurn.attackActionStarted = true;
  actualTurn.attackRollsRemaining = Math.max(0, actualTurn.attackRollsRemaining - 1);
  if (actualTurn.attackRollsRemaining === 0) {
    active.hasActed = true;
  }
}

function applyRandomRayAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'random_ray' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  const target = state.creatures.find((creature) => creature.id === action.targetId);
  const rays = getActiveActions(active).filter((candidate) =>
    active.monsterData.name === 'Beholder' &&
    candidate.name.includes('Ray') &&
    candidate.name !== 'Eye Rays' &&
    candidate.legendaryOnly !== true
  );
  if (!target || !target.isAlive || target.team === active.team || rays.length === 0 || !canSee(state, active, target)) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  const reachableRays = rays.filter((ray) =>
    creatureDistance(active, target) <= (ray.range?.long ?? ray.reach ?? 5)
  );
  if (reachableRays.length === 0) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  const selectedRay = reachableRays[Math.floor(battlecastRandom() * reachableRays.length)];
  if (selectedRay.savingThrow?.area) {
    resolveAoE(state, active, selectedRay, [target]);
  } else if (selectedRay.savingThrow) {
    resolveSingleTargetSave(state, active, target, selectedRay);
  } else {
    resolveAttack(state, active, target, selectedRay);
  }

  actualTurn.attackActionStarted = true;
  actualTurn.attackRollsRemaining = Math.max(0, actualTurn.attackRollsRemaining - 1);
  if (actualTurn.attackRollsRemaining === 0) {
    active.hasActed = true;
  }
}

function applySmiteAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'smite' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  const pending = actualTurn.pendingSmite;
  if (!pending || action.targetId !== pending.targetId) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  if (action.smite === 'decline') {
    actualTurn.pendingSmite = undefined;
    pushLog(state, {
      round: state.round,
      turn: state.turnIndex,
      actor: active.displayName,
      action: 'Divine Smite',
      details: `${active.displayName} does not spend a Divine Smite resource on this hit.`,
      type: 'special',
    });
    return;
  }

  const target = state.creatures.find((creature) => creature.id === pending.targetId);
  const battlecastAction = getActiveActions(active).find((candidate) =>
    candidate.name === pending.actionName && candidate.smiteOnHit
  );
  const choice = smiteChoiceFromAction(action);
  if (!target || !battlecastAction || !choice || !resolveDivineSmite(state, active, target, battlecastAction, choice, pending.isCritical)) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }
  actualTurn.pendingSmite = undefined;
}

function smiteChoiceFromAction(action: Extract<LegalAction, { type: 'smite' }>): DivineSmiteChoice | undefined {
  if (action.smite !== 'divine_smite' || !action.resourceKey || !action.slotLevel) return undefined;
  return {
    resourceKey: action.resourceKey,
    slotLevel: action.slotLevel,
    freeUse: action.resourceKey === 'free-divine-smite',
  };
}

function applyDodgeAction(
  state: BattleState,
  active: Creature,
  actualTurn: ActualTurnContext,
): void {
  active.turnFlags = {
    ...active.turnFlags,
    dodge: true,
  };
  active.hasActed = true;
  actualTurn.attackRollsRemaining = 0;
  active.stats.actionUsage.Dodge = (active.stats.actionUsage.Dodge || 0) + 1;
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Dodge',
    details: `${active.displayName} focuses on defense until their next turn.`,
    type: 'special',
  });
  state.events.push({
    kind: 'effect',
    creatureId: active.id,
    label: 'Dodge',
    tone: 'default',
    durationMs: BASE_DURATIONS.effect,
  });
}

function applyHelpAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'help' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  const target = state.creatures.find((creature) => creature.id === action.targetId);
  if (!target || !target.isAlive || target.team === active.team || creatureDistance(active, target) > 5) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  const key = `help:${active.id}:${target.id}`;
  target.activeBuffs = (target.activeBuffs ?? []).filter((buff) => buff.key !== key);
  target.activeBuffs.push({
    name: 'Help',
    key,
    casterId: active.id,
    appliedRound: state.round,
    endRound: state.round + 2,
    advantageForAllAttackers: true,
    expiresOnSourceTurnStart: true,
  });
  active.hasActed = true;
  actualTurn.attackRollsRemaining = 0;
  active.stats.actionUsage.Help = (active.stats.actionUsage.Help || 0) + 1;
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Help',
    details: `${active.displayName} distracts ${target.displayName}, giving the next attack against it Advantage.`,
    type: 'special',
  });
  state.events.push({
    kind: 'effect',
    creatureId: target.id,
    label: 'Help',
    tone: 'success',
    durationMs: BASE_DURATIONS.effect,
  });
}

function applyStabiliseAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'stabilise' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  const target = state.creatures.find((creature) => creature.id === action.targetId);
  if (!target || !stabiliseDyingAlly(state, active, target)) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }
  active.hasActed = true;
  actualTurn.attackRollsRemaining = 0;
  actualTurn.pendingSmite = undefined;
}

function applyEscapeContainerAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'escape_container' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  if (!active.containedBy || active.containedBy.sourceId !== action.sourceId || !tryEscapeContainer(state, active)) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }
  active.hasActed = true;
  actualTurn.attackRollsRemaining = 0;
  actualTurn.pendingSmite = undefined;
}

function applyLinkedBonusDamageAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'linked_bonus_damage' }>,
  agent: Agent,
): void {
  const target = state.creatures.find((creature) => creature.id === action.targetId);
  if (!target?.isAlive) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }
  const linked = target.activeBuffs?.some((buff) =>
    buff.key === action.buffKey &&
    buff.casterId === active.id &&
    buff.bonusActionDamage &&
    buff.appliedRound < state.round &&
    creatureDistance(active, target) <= (buff.bonusActionDamageRange ?? Infinity)
  );
  if (!linked || !tryUseBonusActionDamageBuff(state, active)) {
    pushInvalidActionLog(state, active, agent, action.id);
  }
}

function applySpellRetargetAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'spell_retarget' }>,
  agent: Agent,
): void {
  const oldTarget = state.creatures.find((creature) => creature.id === action.oldTargetId);
  const target = state.creatures.find((creature) => creature.id === action.targetId);
  if (
    action.spellName !== 'Hex' ||
    action.buffKey !== 'hex' ||
    !oldTarget ||
    !target ||
    !retargetHex(state, active, oldTarget, target)
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
  }
}

async function processPostMoveEffects(
  input: {
    state: BattleState;
    active: Creature;
    actualTurn: ActualTurnContext;
    controllers: MatchControllers;
    agentRng: ReturnType<typeof createRng>;
    battleRng: ReturnType<typeof createRng>;
    replay: ReplayEvent[];
    matchId: string;
    turnStep: number;
    traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
  },
  before: { x: number; y: number },
): Promise<void> {
  const { state, active, actualTurn } = input;
  if (before.x === active.position.x && before.y === active.position.y) return;
  if (!actualTurn.disengaged) {
    const reactions = await prepareOpportunityReactionChoicesForMove(input, before);
    const hooks = createOpportunityAttackHooks({
      state,
      replay: input.replay,
      matchId: input.matchId,
      controllers: input.controllers,
      decisions: reactions,
    });
    await runOpportunityAttacksWithDynamicDamageReactions({
      state,
      active,
      before,
      opportunityAttacks: hooks,
      replay: input.replay,
      matchId: input.matchId,
      controllers: input.controllers,
      agentRng: input.agentRng,
      battleRng: input.battleRng,
      traceSink: input.traceSink,
    });
  }
  if (active.isAlive && !state.isComplete) {
    checkAuraEntry(state, active, before);
  }
}

async function runOpportunityAttacksWithDynamicDamageReactions(input: {
  state: BattleState;
  active: Creature;
  before: { x: number; y: number };
  opportunityAttacks: OpportunityAttackHooks;
  replay: ReplayEvent[];
  matchId: string;
  controllers: MatchControllers;
  agentRng: ReturnType<typeof createRng>;
  battleRng: ReturnType<typeof createRng>;
  traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
}): Promise<void> {
  const decisions: DamageReactionMap = new Map();
  const maxReactionPrompts = 20;
  for (let attempt = 0; attempt <= maxReactionPrompts; attempt += 1) {
    const stateSnapshot = cloneBattleState(input.state);
    const replayLength = input.replay.length;
    const rngSnapshot = input.battleRng.snapshot();
    const active = input.state.creatures.find((creature) => creature.id === input.active.id);
    if (!active) {
      throw new Error(`Active creature ${input.active.id} disappeared during ${input.matchId}`);
    }
    const damageReactions = createDamageReactionHooks({
      state: input.state,
      replay: input.replay,
      matchId: input.matchId,
      controllers: input.controllers,
      decisions,
    });
    const previousDamageReactionHooks = input.state.damageReactionHooks;
    input.state.damageReactionHooks = damageReactions;
    try {
      runOpportunityAttacks(input.state, active, input.before, input.opportunityAttacks);
      return;
    } catch (error) {
      if (!(error instanceof PendingDamageReactionDecision)) throw error;
      const prepared = await chooseDamageReactionAtTrigger({
        state: input.state,
        request: error.request,
        agentRng: input.agentRng,
        matchId: input.matchId,
        traceSink: input.traceSink,
      });
      decisions.set(error.request.triggerKey, prepared);
      restoreBattleState(input.state, stateSnapshot);
      input.replay.length = replayLength;
      input.battleRng.restore(rngSnapshot);
    } finally {
      input.state.damageReactionHooks = previousDamageReactionHooks;
    }
  }

  throw new Error(`Exceeded ${maxReactionPrompts} pending movement damage reaction prompts in ${input.matchId}`);
}

async function prepareOpportunityReactionChoicesForMove(
  input: {
    state: BattleState;
    active: Creature;
    controllers: MatchControllers;
    agentRng: ReturnType<typeof createRng>;
    matchId: string;
    turnStep?: number;
    traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
  },
  before: { x: number; y: number },
): Promise<OpportunityReactionMap> {
  const triggers = opportunityReactionTriggers(input.state, input.active, before);
  const decisions: OpportunityReactionMap = new Map();
  for (const trigger of triggers) {
    const agent = controllerForCreature(trigger.reactor, input.controllers);
    if (!isStepwiseAgent(agent)) continue;
    const decision = await chooseOpportunityReaction({
      state: input.state,
      reactor: trigger.reactor,
      mover: input.active,
      triggerCell: trigger.triggerCell,
      agent,
      agentRng: input.agentRng,
      matchId: input.matchId,
      turnStep: input.turnStep,
      traceSink: input.traceSink,
    });
    if (decision) decisions.set(opportunityReactionKey(trigger.reactor, input.active), decision);
  }
  return decisions;
}

async function chooseDamageReactionAtTrigger(input: {
  state: BattleState;
  request: PendingDamageReactionRequest;
  agentRng: ReturnType<typeof createRng>;
  matchId: string;
  traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
}): Promise<PreparedDamageReaction> {
  const { context, agent } = input.request;
  if (!context.attacker) {
    throw new Error(`Cannot ask ${agent.id} for ${context.reaction} without an attacker`);
  }
  const reactor = damageReactionActor(context);
  const catalogue = context.reaction === 'cutting_words_attack' || context.reaction === 'cutting_words_damage'
    ? generateCuttingWordsReactionActions(
        reactor,
        context.attacker,
        context.reaction === 'cutting_words_attack' ? 'attack' : 'damage',
        {
          incomingDamage: context.incomingDamage,
          damageType: context.damageType,
          attackRollTotal: context.attackRollTotal,
          targetAc: context.targetAc,
          maxRollReduction: context.maxRollReduction,
          expectedRollReduction: context.expectedRollReduction,
        },
      )
    : generateDamageReactionActions(context.target, context.attacker, context.reaction, {
        incomingDamage: context.incomingDamage,
        damageType: context.damageType,
      });
  const selection = await chooseStepwiseAction(agent, {
    state: input.state,
    activeCreature: reactor,
    catalogue,
    rng: input.agentRng,
    traceMeta: {
      matchId: input.matchId,
      round: input.state.round,
      turnIndex: input.state.turnIndex,
      activeCreatureId: reactor.id,
      activeCreatureName: reactor.displayName,
      agentId: agent.id,
    },
    traceSink: input.traceSink,
  });
  if (selection.acceptedAction.type !== 'reaction') {
    throw new Error(`${agent.id} selected non-reaction action ${selection.acceptedAction.id} for ${context.reaction}`);
  }
  return {
    agent,
    catalogue,
    requestedActionId: selection.requestedActionId,
    acceptedAction: selection.acceptedAction,
    llmTrace: selection.trace,
  };
}

async function chooseOpportunityReaction(input: {
  state: BattleState;
  reactor: Creature;
  mover: Creature;
  triggerCell: { x: number; y: number };
  agent: StepwiseAgent;
  agentRng: ReturnType<typeof createRng>;
  matchId: string;
  turnStep?: number;
  traceSink?: (trace: OpenRouterRawDecisionTrace) => void | Promise<void>;
}): Promise<PreparedOpportunityReaction | undefined> {
  const originalPosition = { ...input.mover.position };
  input.mover.position = { ...input.triggerCell };
  try {
    const catalogue = generateOpportunityReactionActions(input.reactor, input.mover);
    let selection: StepwiseSelection;
    try {
      selection = await chooseStepwiseAction(input.agent, {
        state: input.state,
        activeCreature: input.reactor,
        catalogue,
        rng: input.agentRng,
        traceMeta: {
          matchId: input.matchId,
          round: input.state.round,
          turnIndex: input.state.turnIndex,
          turnStep: input.turnStep,
          activeCreatureId: input.reactor.id,
          activeCreatureName: input.reactor.displayName,
          agentId: input.agent.id,
        },
        traceSink: input.traceSink,
      });
    } catch (error) {
      pushLog(input.state, {
        round: input.state.round,
        turn: input.state.turnIndex,
        actor: input.reactor.displayName,
        action: 'Opportunity Attack',
        details: `${input.reactor.displayName} declines the opportunity attack against ${input.mover.displayName} because the controlling model failed to answer (${error instanceof Error ? error.message.slice(0, 120) : 'unknown error'}).`,
        type: 'info',
      });
      return undefined;
    }
    if (selection.acceptedAction.type !== 'reaction') return undefined;
    return {
      agent: input.agent,
      catalogue,
      requestedActionId: selection.requestedActionId,
      acceptedAction: selection.acceptedAction,
      llmTrace: selection.trace,
    };
  } finally {
    input.mover.position = originalPosition;
  }
}

function createOpportunityAttackHooks(input: {
  state: BattleState;
  replay: ReplayEvent[];
  matchId: string;
  controllers: MatchControllers;
  decisions: OpportunityReactionMap;
  missingDecision?: 'decline' | 'request';
}): OpportunityAttackHooks {
  const started = new WeakSet<OpportunityAttackDecisionContext>();
  return {
    chooseOpportunityAttack: (context) => {
      const controller = controllerForCreature(context.reactor, input.controllers);
      if (!isStepwiseAgent(controller)) return undefined;
      const decision = input.decisions.get(opportunityReactionKey(context.reactor, context.mover));
      if (!decision) {
        if (input.missingDecision === 'request') {
          throw new PendingOpportunityReactionDecision({
            reactorId: context.reactor.id,
            moverId: context.mover.id,
            triggerCell: { ...context.triggerCell },
          });
        }
        pushLog(input.state, {
          round: input.state.round,
          turn: input.state.turnIndex,
          actor: context.reactor.displayName,
          action: 'Opportunity Attack',
          details: `${context.reactor.displayName} declines the opportunity attack against ${context.mover.displayName} because no model decision was available for this trigger.`,
          type: 'info',
        });
        return 'decline';
      }
      if (decision.acceptedAction.reaction === 'decline') return 'decline';
      return context.meleeActions.find((action) => action.name === decision.acceptedAction.actionName) ?? 'decline';
    },
    beforeOpportunityAttack: (context, _action) => {
      const decision = input.decisions.get(opportunityReactionKey(context.reactor, context.mover));
      if (!decision || started.has(context)) return undefined;
      started.add(context);
      input.replay.push({
        type: 'turn_started',
        matchId: input.matchId,
        round: input.state.round,
        turnIndex: input.state.turnIndex,
        activeCreatureId: context.reactor.id,
        activeCreatureName: context.reactor.displayName,
        controller: describeAgentController(decision.agent),
        actionSpace: decision.catalogue.actionSpace,
        legalActions: decision.catalogue.actions,
        actionEconomy: decision.catalogue.actionEconomy,
        stateHash: hashBattlecastState(input.state),
      });
      return {
        logsBefore: input.state.logs.length,
        eventsBefore: input.state.events.length,
      };
    },
    afterOpportunityAttack: (context, _action, before) => {
      const decision = input.decisions.get(opportunityReactionKey(context.reactor, context.mover));
      if (!decision || !before) return;
      input.replay.push({
        type: 'action_resolved',
        matchId: input.matchId,
        round: input.state.round,
        turnIndex: input.state.turnIndex,
        activeCreatureId: context.reactor.id,
        agentId: decision.agent.id,
        requestedActionId: decision.requestedActionId,
        acceptedAction: decision.acceptedAction,
        llmTrace: decision.llmTrace,
        logs: input.state.logs.slice(before.logsBefore),
        events: input.state.events.slice(before.eventsBefore),
        stateHash: hashBattlecastState(input.state),
      });
    },
  };
}

function createDamageReactionHooks(input: {
  state: BattleState;
  replay: ReplayEvent[];
  matchId: string;
  controllers: MatchControllers;
  decisions: DamageReactionMap;
}): DamageReactionHooks {
  const started = new WeakSet<DamageReactionDecisionContext>();
  const triggerKeys = new WeakMap<DamageReactionDecisionContext, string>();
  let triggerIndex = 0;
  return {
    chooseDamageReaction: (context) => {
      const reactor = damageReactionActor(context);
      const controller = controllerForCreature(reactor, input.controllers);
      if (!isStepwiseAgent(controller)) return undefined;
      const attacker = context.attacker;
      if (!attacker) return 'decline';
      const triggerKey = damageReactionTriggerKey(context, triggerIndex);
      triggerIndex += 1;
      triggerKeys.set(context, triggerKey);
      const decision = input.decisions.get(triggerKey);
      if (!decision) {
        throw new PendingDamageReactionDecision({ triggerKey, context, agent: controller });
      }
      return decision.acceptedAction.reaction === context.reaction ? 'use' : 'decline';
    },
    beforeDamageReaction: (context, _decision) => {
      const triggerKey = triggerKeys.get(context);
      if (!triggerKey) return undefined;
      const prepared = input.decisions.get(triggerKey);
      if (!prepared || started.has(context)) return undefined;
      started.add(context);
      input.replay.push({
        type: 'turn_started',
        matchId: input.matchId,
        round: input.state.round,
        turnIndex: input.state.turnIndex,
        activeCreatureId: damageReactionActor(context).id,
        activeCreatureName: damageReactionActor(context).displayName,
        controller: describeAgentController(prepared.agent),
        actionSpace: prepared.catalogue.actionSpace,
        legalActions: prepared.catalogue.actions,
        actionEconomy: prepared.catalogue.actionEconomy,
        stateHash: hashBattlecastState(input.state),
      });
      return {
        logsBefore: input.state.logs.length,
        eventsBefore: input.state.events.length,
      };
    },
    afterDamageReaction: (context, decision, before) => {
      const triggerKey = triggerKeys.get(context);
      if (!triggerKey) return;
      const prepared = input.decisions.get(triggerKey);
      if (!prepared || !before) return;
      const acceptedAction = {
        ...prepared.acceptedAction,
        incomingDamage: context.incomingDamage,
        damageType: context.damageType,
        expectedDamageReduction: decision === 'use' ? prepared.acceptedAction.expectedDamageReduction : 0,
        actualDamageReduction: decision === 'use' ? context.actualDamageReduction : 0,
        attackRollTotal: context.attackRollTotal,
        targetAc: context.targetAc,
        maxRollReduction: context.maxRollReduction,
        expectedRollReduction: context.expectedRollReduction,
        actualRollReduction: decision === 'use' ? context.actualRollReduction : 0,
        preventedHit: decision === 'use' ? context.preventedHit : false,
      };
      input.replay.push({
        type: 'action_resolved',
        matchId: input.matchId,
        round: input.state.round,
        turnIndex: input.state.turnIndex,
        activeCreatureId: damageReactionActor(context).id,
        agentId: prepared.agent.id,
        requestedActionId: prepared.requestedActionId,
        acceptedAction,
        llmTrace: prepared.llmTrace,
        logs: input.state.logs.slice(before.logsBefore),
        events: input.state.events.slice(before.eventsBefore),
        stateHash: hashBattlecastState(input.state),
      });
    },
  };
}

function damageReactionActor(context: DamageReactionDecisionContext): Creature {
  return context.reactor ?? context.target;
}

function opportunityReactionTriggers(
  state: BattleState,
  mover: Creature,
  oldPos: { x: number; y: number },
): Array<{ reactor: Creature; triggerCell: { x: number; y: number } }> {
  const moveEvent = [...state.events].reverse().find((event) =>
    event.kind === 'move' && event.creatureId === mover.id
  );
  const movePath = moveEvent?.kind === 'move' && moveEvent.path
    ? moveEvent.path
    : [oldPos, mover.position];
  return state.creatures.flatMap((reactor) => {
    if (reactor.team === mover.team || !canMakeOpportunityAttack(reactor, mover)) return [];
    const meleeActions = getMeleeActions(reactor);
    const reach = meleeActions.reduce((max, action) => Math.max(max, action.reach || 5), 5);
    let triggerCell: { x: number; y: number } | null = null;
    for (const cell of movePath) {
      const cellDistance = Math.max(
        Math.abs(reactor.position.x - cell.x),
        Math.abs(reactor.position.y - cell.y),
      ) * 5;
      if (cellDistance <= reach) triggerCell = cell;
    }
    if (!triggerCell || creatureDistance(reactor, mover) <= reach) return [];
    const moverIsAirborne = !!mover.airborne;
    const reactorIsFlyer = (getActiveSpeed(reactor).fly ?? 0) > 0;
    if (moverIsAirborne && !reactorIsFlyer) return [];
    return [{ reactor, triggerCell }];
  });
}

function canMakeOpportunityAttack(reactor: Creature, mover: Creature): boolean {
  if (!reactor.isAlive || reactor.dying || reactor.team === mover.team) return false;
  if (reactor.activeBuffs?.some((buff) => buff.preventsOpportunityAttacks)) return false;
  const reactionLimit = getHydraHeadCount(reactor) ?? 1;
  const reactionsUsed = reactor.reactionsUsed ?? (reactor.reactionUsed ? 1 : 0);
  if (reactionsUsed >= reactionLimit) return false;
  if (reactor.conditions.includes('incapacitated') || reactor.conditions.includes('stunned') ||
      reactor.conditions.includes('paralyzed') || reactor.conditions.includes('unconscious')) return false;
  return getMeleeActions(reactor).length > 0;
}

function opportunityReactionKey(reactor: Creature, mover: Creature): string {
  return `${reactor.id}->${mover.id}`;
}

function damageReactionTriggerKey(context: DamageReactionDecisionContext, triggerIndex: number): string {
  return [
    context.reactor?.id ?? context.target.id,
    context.target.id,
    context.attacker?.id ?? 'no-attacker',
    context.reaction,
    triggerIndex,
  ].join(':');
}

function controllerForCreature(creature: Creature, controllers: MatchControllers): Agent {
  return creature.team === 'red' ? controllers.red : controllers.blue;
}

function applySpellAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'spell' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  const battlecastAction = getActiveActions(active).find((candidate) => candidate.name === action.actionName);
  if (!battlecastAction) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  const targetIds = action.targetIds?.length ? action.targetIds : action.targetId ? [action.targetId] : [];
  const targets = targetIds
    .map((targetId) => state.creatures.find((creature) => creature.id === targetId))
    .filter((target): target is Creature => !!target);
  const primaryTarget = action.targetId
    ? state.creatures.find((creature) => creature.id === action.targetId) ?? targets[0] ?? null
    : targets[0] ?? null;
  let applied = false;

  if (battlecastAction.name === "Nature's Veil") {
    applyNaturesVeilAction(state, active, action, agent);
    return;
  }

  if (battlecastAction.name === 'Swallow' && primaryTarget) {
    applied = resolveSwallowAction(state, active, primaryTarget, battlecastAction);
  } else if (battlecastAction.autoDarts) {
    applied = executeSpell(state, active, battlecastAction, primaryTarget, targets);
  } else if (battlecastAction.spellLevel !== undefined || battlecastAction.resourceCost || battlecastAction.heal || battlecastAction.temporaryHp || battlecastAction.buff || battlecastAction.powerWord) {
    const aoeTargets = battlecastAction.savingThrow?.area || (battlecastAction.targetScope === 'area_enemies' && targets.length > 1)
      ? targets.length > 0 ? targets : getAoETargets(state, active, battlecastAction)
      : undefined;
    applied = executeSpell(state, active, battlecastAction, primaryTarget, aoeTargets, action.center, action.direction);
  } else if (battlecastAction.savingThrow?.area) {
    const aoeTargets = targets.length > 0 ? targets : getAoETargets(state, active, battlecastAction);
    resolveAoE(state, active, battlecastAction, aoeTargets, action.center, action.direction);
    if (battlecastAction.recharge) active.recharges[battlecastAction.name] = false;
    applied = true;
  } else if (battlecastAction.savingThrow && primaryTarget) {
    resolveSingleTargetSave(state, active, primaryTarget, battlecastAction);
    if (battlecastAction.recharge) active.recharges[battlecastAction.name] = false;
    applied = true;
  } else if (battlecastAction.attackBonus !== undefined && primaryTarget) {
    resolveAttack(state, active, primaryTarget, battlecastAction);
    applied = true;
  }

  if (!applied) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  applyInstinctivePounce(state, active, battlecastAction);

  if (battlecastAction.isBonusAction) {
    active.bonusActionUsed = true;
  } else {
    active.hasActed = true;
    actualTurn.attackRollsRemaining = 0;
  }
}

function applyNaturesVeilAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'spell' }>,
  agent: Agent,
): void {
  const resourceCost = action.resourceCost ?? { key: 'natures-veil', amount: 1 };
  if (
    active.monsterData.heroClass !== 'Ranger' ||
    (active.monsterData.heroLevel ?? 0) < 14 ||
    active.bonusActionUsed ||
    active.conditions.includes('invisible') ||
    action.targetId !== active.id ||
    !hasResource(active, resourceCost.key, resourceCost.amount)
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  consumeResource(active, resourceCost.key, resourceCost.amount);
  active.bonusActionUsed = true;
  active.conditions.push('invisible');
  active.conditionTimers.push({
    condition: 'invisible',
    duration: 'end_of_next_turn',
    appliedRound: state.round,
    sourceId: active.id,
  });
  active.stats.actionUsage["Nature's Veil"] = (active.stats.actionUsage["Nature's Veil"] || 0) + 1;
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Invisible',
    details: `${active.displayName} is now invisible!`,
    type: 'condition',
  });
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: "Nature's Veil",
    details: `${active.displayName} uses a bonus action to become Invisible until the end of their next turn.`,
    type: 'special',
  });
  state.events.push({
    kind: 'condition',
    creatureId: active.id,
    condition: 'invisible',
    applied: true,
    durationMs: BASE_DURATIONS.condition,
  });
}

function applyInstinctivePounce(
  state: BattleState,
  active: Creature,
  action: { name: string },
): void {
  if (action.name !== 'Rage') return;
  if (active.monsterData.heroClass !== 'Barbarian' || (active.monsterData.heroLevel ?? 0) < 7) return;
  const target = state.creatures
    .filter((creature) => creature.team !== active.team && creature.isAlive)
    .sort((left, right) =>
      creatureDistance(active, left) - creatureDistance(active, right) ||
      left.id.localeCompare(right.id)
    )[0];
  if (!target) return;

  const oldRemaining = active.movementRemaining;
  const from = { ...active.position };
  active.movementRemaining = Math.floor(getEffectiveMoveSpeed(active, state) / 2);
  active.position = moveToward(active, target.position, state);
  active.movementRemaining = oldRemaining;
  const moved = distance(from, active.position);
  if (moved > 0) {
    pushLog(state, {
      round: state.round,
      turn: state.turnIndex,
      actor: active.displayName,
      action: 'Instinctive Pounce',
      details: `${active.displayName} surges ${moved} ft as part of entering Rage.`,
      type: 'move',
    });
  }
}

function applyDashAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'dash' }>,
  actualTurn: ActualTurnContext,
): void {
  const extraMovement = Math.max(0, action.extraMovement || movementAllowance(active, state));
  active.movementRemaining += extraMovement;
  active.hasActed = true;
  actualTurn.attackRollsRemaining = 0;
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Dash',
    details: `${active.displayName} dashes, gaining ${extraMovement} ft of movement for this turn.`,
    type: 'move',
  });
}

function applyDisengageAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'disengage' }>,
  actualTurn: ActualTurnContext,
): void {
  actualTurn.disengaged = true;
  if (action.isBonusAction) {
    active.bonusActionUsed = true;
  } else {
    active.hasActed = true;
    actualTurn.attackRollsRemaining = 0;
  }
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: action.isBonusAction ? 'Bonus Action Disengage' : 'Disengage',
    details: `${active.displayName} avoids opportunity attacks from movement this turn.`,
    type: 'move',
  });
  for (const enemy of opportunityThreats(state, active)) {
    state.events.push({
      kind: 'oaAvoided',
      moverId: active.id,
      enemyId: enemy.id,
      reason: action.isBonusAction ? bonusDisengageReason(active) : 'disengage',
      durationMs: BASE_DURATIONS.oaAvoided,
    });
  }
}

function applyClassFeatureAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  if (action.feature === 'steady_aim') {
    active.turnFlags = {
      ...active.turnFlags,
      steadyAim: true,
    };
    active.bonusActionUsed = true;
    active.movementRemaining = 0;
    active.stats.actionUsage['Steady Aim'] = (active.stats.actionUsage['Steady Aim'] || 0) + 1;
    pushLog(state, {
      round: state.round,
      turn: state.turnIndex,
      actor: active.displayName,
      action: 'Steady Aim',
      details: `${active.displayName} holds position and gains Advantage on the next attack.`,
      type: 'special',
    });
    state.events.push({
      kind: 'effect',
      creatureId: active.id,
      label: 'Steady Aim',
      tone: 'success',
      durationMs: BASE_DURATIONS.effect,
    });
    return;
  }

  if (action.feature === 'action_surge') {
    applyActionSurgeAction(state, active, action, agent, actualTurn);
    return;
  }

  if (action.feature === 'sacred_weapon') {
    applySacredWeaponAction(state, active, action, agent, actualTurn);
    return;
  }

  if (action.feature === 'superior_defense') {
    applySuperiorDefenseAction(state, active, action, agent, actualTurn);
    return;
  }

  if (action.feature === 'reckless_attack') {
    applyRecklessAttackAction(state, active, action, agent, actualTurn);
    return;
  }

  if (action.feature === 'brutal_strike') {
    applyBrutalStrikeAction(state, active, action, agent, actualTurn);
    return;
  }

  if (action.feature === 'quivering_palm') {
    applyQuiveringPalmAction(state, active, action, agent, actualTurn);
    return;
  }

  if (action.feature === 'wild_shape') {
    applyWildShapeAction(state, active, action, agent);
    return;
  }

  if (action.feature === 'frenzy') {
    applyFrenzyAction(state, active, action, agent);
    return;
  }

  if (action.feature === 'martial_arts_strike') {
    applyMonkBonusStrike(state, active, action, agent, actualTurn, 'martial_arts_strike');
    return;
  }

  if (action.feature === 'flurry_of_blows') {
    applyMonkBonusStrike(state, active, action, agent, actualTurn, 'flurry_of_blows');
  }
}

function applySacredWeaponAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  if (
    active.monsterData.heroClass !== 'Paladin' ||
    (active.monsterData.heroLevel ?? 0) < 3 ||
    active.hasActed ||
    actualTurn.attackActionStarted ||
    !hasResource(active, 'channel-divinity') ||
    hasBuff(active, 'sacred-weapon') ||
    !hasReachableMeleeTarget(state, active)
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  const attackBonus = Math.max(1, abilityModifier(active.monsterData.abilities.cha));
  consumeResource(active, 'channel-divinity');
  addBuff(active, {
    name: 'Sacred Weapon',
    key: 'sacred-weapon',
    casterId: active.id,
    appliedRound: state.round,
    endRound: state.round + 100,
    attackBonus,
  });
  active.stats.actionUsage['Sacred Weapon'] = (active.stats.actionUsage['Sacred Weapon'] || 0) + 1;
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Sacred Weapon',
    details: `${active.displayName} empowers their weapon, adding +${attackBonus} to melee attack rolls.`,
    type: 'special',
  });
  state.events.push({
    kind: 'effect',
    creatureId: active.id,
    label: 'Sacred Weapon',
    tone: 'success',
    durationMs: BASE_DURATIONS.effect,
  });
}

function applySuperiorDefenseAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  if (
    active.monsterData.heroClass !== 'Monk' ||
    (active.monsterData.heroLevel ?? 0) < 18 ||
    active.hasActed ||
    actualTurn.attackActionStarted ||
    active.hasMovedThisTurn ||
    !hasResource(active, 'ki', 3) ||
    hasBuff(active, 'superior-defense') ||
    active.conditions.includes('incapacitated') ||
    active.conditions.includes('stunned') ||
    active.conditions.includes('paralyzed') ||
    active.conditions.includes('petrified') ||
    active.conditions.includes('unconscious') ||
    !state.creatures.some((target) => target.team !== active.team && target.isAlive && !target.dying)
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  consumeResource(active, 'ki', 3);
  addBuff(active, {
    name: 'Superior Defense',
    key: 'superior-defense',
    casterId: active.id,
    appliedRound: state.round,
    endRound: state.round + 10,
    resistAllDamageExcept: ['force'],
  });
  active.stats.actionUsage['Superior Defense'] = (active.stats.actionUsage['Superior Defense'] || 0) + 1;
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Superior Defense',
    details: `${active.displayName} spends 3 Focus Points for resistance to all damage except Force.`,
    type: 'special',
  });
  state.events.push({
    kind: 'effect',
    creatureId: active.id,
    label: 'Superior Defense',
    tone: 'success',
    durationMs: BASE_DURATIONS.effect,
  });
}

function applyRecklessAttackAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  const meleeTargetsAvailable = getActiveActions(active)
    .filter((candidate) => candidate.type === 'melee' && candidate.attackBonus !== undefined && candidate.legendaryOnly !== true)
    .some((candidate) => state.creatures.some((target) =>
      target.team !== active.team &&
      target.isAlive &&
      !target.dying &&
      creatureDistance(active, target) <= (candidate.reach ?? 5)
    ));
  if (
    active.monsterData.heroClass !== 'Barbarian' ||
    (active.monsterData.heroLevel ?? 0) < 2 ||
    active.hasActed ||
    actualTurn.attackActionStarted ||
    active.turnFlags?.reckless ||
    active.turnFlags?.brutalStrike ||
    !meleeTargetsAvailable
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  active.turnFlags = {
    ...active.turnFlags,
    reckless: true,
  };
  active.stats.actionUsage['Reckless Attack'] = (active.stats.actionUsage['Reckless Attack'] || 0) + 1;
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Reckless Attack',
    details: `${active.displayName} attacks recklessly, gaining Advantage on melee attacks this turn while attacks against them have Advantage until their next turn.`,
    type: 'special',
  });
  state.events.push({
    kind: 'effect',
    creatureId: active.id,
    label: 'Reckless Attack',
    tone: 'success',
    durationMs: BASE_DURATIONS.effect,
  });
}

function applyBrutalStrikeAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  const meleeTargetsAvailable = getActiveActions(active)
    .filter((candidate) => candidate.type === 'melee' && candidate.attackBonus !== undefined && candidate.legendaryOnly !== true)
    .some((candidate) => state.creatures.some((target) =>
      target.team !== active.team &&
      target.isAlive &&
      !target.dying &&
      creatureDistance(active, target) <= (candidate.reach ?? 5)
    ));
  if (
    active.monsterData.heroClass !== 'Barbarian' ||
    (active.monsterData.heroLevel ?? 0) < 9 ||
    active.hasActed ||
    actualTurn.attackActionStarted ||
    active.turnFlags?.reckless ||
    active.turnFlags?.brutalStrike ||
    active.turnFlags?.brutalStrikeUsed ||
    !meleeTargetsAvailable
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  active.turnFlags = {
    ...active.turnFlags,
    brutalStrike: true,
  };
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Brutal Strike Declared',
    details: `${active.displayName} forgoes Reckless Attack advantage to channel Brutal Strike on the next melee hit this turn.`,
    type: 'special',
  });
  state.events.push({
    kind: 'effect',
    creatureId: active.id,
    label: 'Brutal Strike',
    tone: 'success',
    durationMs: BASE_DURATIONS.effect,
  });
}

function applyQuiveringPalmAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  const target = action.targetId
    ? state.creatures.find((creature) => creature.id === action.targetId)
    : undefined;
  const key = `quivering-palm:${active.id}`;
  if (
    active.monsterData.heroClass !== 'Monk' ||
    (active.monsterData.heroLevel ?? 0) < 17 ||
    active.hasActed ||
    actualTurn.attackActionStarted ||
    !target ||
    target.team === active.team ||
    !target.isAlive ||
    target.dying ||
    !target.activeBuffs?.some((buff) => buff.key === key)
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  target.activeBuffs = target.activeBuffs.filter((buff) => buff.key !== key);
  const wisMod = abilityModifier(active.monsterData.abilities.wis);
  const dc = 8 + active.monsterData.proficiencyBonus + wisMod;
  const saveMod = getEffectiveSaveModifier(target, 'con', state);
  const save = rollSaveWithBuffs(target, saveMod, false, dc, 'con');
  const success = save.total >= dc;
  const rawDamage = rollDice('10d12').total;
  const damage = success ? Math.floor(rawDamage / 2) : rawDamage;

  state.events.push({
    kind: 'attack',
    attackerId: active.id,
    targetId: target.id,
    actionName: 'Quivering Palm',
    attackType: 'touch',
    durationMs: BASE_DURATIONS.attack,
  });
  state.events.push({ kind: 'save', targetId: target.id, success, durationMs: BASE_DURATIONS.save });
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Quivering Palm',
    details: `${active.displayName} ends Quivering Palm on ${target.displayName}; ${target.displayName} ${success ? 'resists' : 'fails'} (${save.total} vs DC ${dc}) and takes ${damage} force damage.`,
    damage,
    type: 'damage',
  });
  const before = target.currentHp;
  state.events.push({
    kind: 'hit',
    targetId: target.id,
    damage,
    damageType: 'force',
    critical: false,
    targetHpBefore: before,
    targetHpAfter: before,
    durationMs: BASE_DURATIONS.hit,
  });
  const hitEvent = state.events[state.events.length - 1];
  applyDamage(state, target, damage, 'force', active, false, true, false);
  if (hitEvent.kind === 'hit') hitEvent.targetHpAfter = target.currentHp;
  active.stats.actionUsage['Quivering Palm'] = (active.stats.actionUsage['Quivering Palm'] || 0) + 1;
  active.hasActed = true;
  actualTurn.attackRollsRemaining = 0;
  actualTurn.pendingSmite = undefined;
}

function hasReachableMeleeTarget(state: BattleState, active: Creature): boolean {
  return getActiveActions(active)
    .filter((candidate) => candidate.type === 'melee' && candidate.attackBonus !== undefined && candidate.legendaryOnly !== true)
    .some((candidate) => state.creatures.some((target) =>
      target.team !== active.team &&
      target.isAlive &&
      !target.dying &&
      creatureDistance(active, target) <= (candidate.reach ?? 5)
    ));
}

function applyActionSurgeAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
): void {
  if (
    active.monsterData.heroClass !== 'Fighter' ||
    (active.monsterData.heroLevel ?? 0) < 2 ||
    !hasResource(active, 'action-surge') ||
    (!active.hasActed && !actualTurn.attackActionStarted)
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  consumeResource(active, 'action-surge');
  active.hasActed = false;
  actualTurn.attackActionStarted = false;
  actualTurn.attackRollsRemaining = estimateAttackRollBudget(active);
  active.stats.actionUsage['Action Surge'] = (active.stats.actionUsage['Action Surge'] || 0) + 1;
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Action Surge',
    details: `${active.displayName} uses Action Surge for an extra action.`,
    type: 'special',
  });
  state.events.push({
    kind: 'effect',
    creatureId: active.id,
    label: 'Action Surge',
    tone: 'success',
    durationMs: BASE_DURATIONS.effect,
  });
}

function applyWildShapeAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
): void {
  const level = active.monsterData.heroLevel ?? 0;
  if (
    active.monsterData.heroClass !== 'Druid' ||
    level < 2 ||
    active.wildShape ||
    active.bonusActionUsed ||
    active.concentratingOn ||
    !hasResource(active, 'wild-shape') ||
    !action.beastName
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  const beast = getEligibleWildShapeBeasts({
    level,
    subclass: active.monsterData.heroSubclass,
  }).find((candidate) => candidate.name === action.beastName);
  if (!beast || !canWildShapeFit(state, active, beast.size)) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  consumeResource(active, 'wild-shape');
  active.bonusActionUsed = true;
  const isMoon = active.monsterData.heroSubclass === 'Circle of the Moon';
  const tempHp = isMoon ? level * 3 : level;
  const moonAc = 13 + abilityModifier(active.monsterData.abilities.wis);
  const ac = isMoon ? Math.max(beast.ac, moonAc) : beast.ac;
  active.wildShape = {
    beastName: beast.name,
    tempHp,
    maxTempHp: tempHp,
    formHp: beast.formHp,
    cr: beast.cr,
    ac,
    speed: beast.speed,
    actions: beast.actions,
    size: beast.size,
    traits: beast.traits,
    saves: beast.saves,
    abilities: beast.abilities,
    isMoon,
  };
  if (beast.initialResources) {
    for (const [key, value] of Object.entries(beast.initialResources)) {
      active.resources[key] = value;
    }
  }
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Wild Shape',
    details: `${active.displayName} transforms into a ${beast.name}! (${tempHp} temporary HP, AC ${ac})`,
    type: 'special',
  });
  active.stats.actionUsage['Wild Shape'] = (active.stats.actionUsage['Wild Shape'] || 0) + 1;
  state.events.push({ kind: 'wildShape', creatureId: active.id, beastName: beast.name, durationMs: 0 });
}

function applyFrenzyAction(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
): void {
  const target = action.targetId
    ? state.creatures.find((creature) => creature.id === action.targetId)
    : undefined;
  const meleeActions = getActiveActions(active)
    .filter((candidate) => candidate.type === 'melee' && candidate.attackBonus !== undefined && candidate.legendaryOnly !== true);
  const best = target && meleeActions.length > 0
    ? meleeActions.reduce((left, right) =>
      estimateActionDamage(right, target) > estimateActionDamage(left, target) ? right : left
    )
    : undefined;
  if (
    active.monsterData.heroClass !== 'Barbarian' ||
    (active.monsterData.heroLevel ?? 0) < 3 ||
    active.bonusActionUsed ||
    !hasActiveBuff(active, 'rage') ||
    !target ||
    !target.isAlive ||
    !best ||
    creatureDistance(active, target) > (best.reach ?? 5)
  ) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  active.bonusActionUsed = true;
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Frenzy',
    details: `${active.displayName} makes a frenzied bonus attack!`,
    type: 'special',
  });
  active.stats.actionUsage['Frenzy'] = (active.stats.actionUsage['Frenzy'] || 0) + 1;
  resolveAttack(state, active, target, best);
}

function canWildShapeFit(
  state: BattleState,
  active: Creature,
  size: Creature['monsterData']['size'],
): boolean {
  const gridSize = state.gridSize ?? 20;
  const footprint = getFootprintSize(size);
  if (active.position.x + footprint > gridSize || active.position.y + footprint > gridSize) return false;
  return !isPositionBlocked(active.position, size, state.creatures, active.id, state.terrainBlocked);
}

function hasActiveBuff(active: Creature, key: string): boolean {
  return active.activeBuffs?.some((buff) => buff.key === key) ?? false;
}

function applyMonkBonusStrike(
  state: BattleState,
  active: Creature,
  action: Extract<LegalAction, { type: 'class_feature' }>,
  agent: Agent,
  actualTurn: ActualTurnContext,
  feature: 'martial_arts_strike' | 'flurry_of_blows',
): void {
  const target = action.targetId
    ? state.creatures.find((creature) => creature.id === action.targetId)
    : undefined;
  const unarmed = monkUnarmedAction(active);
  if (!target || !target.isAlive || !unarmed || creatureDistance(active, target) > (unarmed.reach ?? 5)) {
    pushInvalidActionLog(state, active, agent, action.id);
    return;
  }

  if (feature === 'martial_arts_strike') {
    if (active.bonusActionUsed) {
      pushInvalidActionLog(state, active, agent, action.id);
      return;
    }
    active.bonusActionUsed = true;
    actualTurn.flurryStrikesRemaining = 0;
    pushLog(state, {
      round: state.round,
      turn: state.turnIndex,
      actor: active.displayName,
      action: 'Martial Arts',
      details: `${active.displayName} follows up with a bonus unarmed strike.`,
      type: 'special',
    });
    active.stats.actionUsage['Martial Arts'] = (active.stats.actionUsage['Martial Arts'] || 0) + 1;
    resolveAttack(state, active, target, unarmed);
    return;
  }

  if (actualTurn.flurryStrikesRemaining <= 0) {
    if (active.bonusActionUsed || !hasResource(active, 'ki')) {
      pushInvalidActionLog(state, active, agent, action.id);
      return;
    }
    active.bonusActionUsed = true;
    consumeResource(active, 'ki');
    actualTurn.flurryStrikesRemaining = Math.max(0, flurryStrikeCount(active) - 1);
    pushLog(state, {
      round: state.round,
      turn: state.turnIndex,
      actor: active.displayName,
      action: 'Flurry of Blows',
      details: `${active.displayName} spends 1 ki for Flurry of Blows!`,
      type: 'special',
    });
    active.stats.actionUsage['Flurry of Blows'] = (active.stats.actionUsage['Flurry of Blows'] || 0) + 1;
  } else {
    actualTurn.flurryStrikesRemaining = Math.max(0, actualTurn.flurryStrikesRemaining - 1);
  }

  withOpenHandFlurryFlag(active, () => {
    resolveAttack(state, active, target, unarmed);
  });
}

function flurryStrikeCount(active: Creature): number {
  return (active.monsterData.heroLevel ?? 0) >= 10 ? 3 : 2;
}

function monkUnarmedAction(active: Creature) {
  const meleeActions = getActiveActions(active)
    .filter((candidate) => candidate.type === 'melee' && candidate.attackBonus !== undefined && candidate.legendaryOnly !== true);
  return meleeActions.find((candidate) => candidate.name === 'Martial Arts (Unarmed)') ?? meleeActions[0];
}

function withOpenHandFlurryFlag(active: Creature, fn: () => void): void {
  const hadOpenHandFlag = active.turnFlags?.openHandFlurryStrike;
  active.turnFlags = {
    ...active.turnFlags,
    openHandFlurryStrike: true,
  };
  try {
    fn();
  } finally {
    if (hadOpenHandFlag) {
      active.turnFlags.openHandFlurryStrike = true;
    } else {
      delete active.turnFlags.openHandFlurryStrike;
    }
  }
}

function shouldEndActualTurn(active: Creature, actualTurn: ActualTurnContext): boolean {
  if (actualTurn.ended) return true;
  if (actualTurn.pendingSmite) return false;
  if (actualTurn.flurryStrikesRemaining > 0) return false;
  const hasMainAction = !active.hasActed || (actualTurn.attackActionStarted && actualTurn.attackRollsRemaining > 0);
  const hasBonusAction = active.bonusActionUsed !== true;
  const hasMovement = active.movementRemaining > 0;
  return !hasMainAction && !hasBonusAction && !hasMovement;
}

function movementAllowance(active: Creature, state: BattleState): number {
  if (active.conditions.includes('restrained') || active.conditions.includes('grappled')) return 0;
  return Math.max(0, getEffectiveMoveSpeed(active, state) - activeSpeedPenalty(active));
}

function activeSpeedPenalty(active: Creature): number {
  return Math.max(0, ...((active.activeBuffs ?? []).map((buff) => buff.speedPenalty ?? 0)));
}

function bonusDisengageReason(active: Creature): 'cunning' | 'nimble' {
  return active.monsterData.heroClass === 'Rogue' ? 'cunning' : 'nimble';
}

function opportunityThreats(state: BattleState, active: Creature): Creature[] {
  return state.creatures.filter((enemy) => {
    if (enemy.team === active.team || !enemy.isAlive || enemy.reactionUsed) return false;
    if (enemy.conditions.includes('incapacitated') || enemy.conditions.includes('stunned') ||
      enemy.conditions.includes('paralyzed') || enemy.conditions.includes('unconscious')) return false;
    const reach = getActiveActions(enemy)
      .filter((enemyAction) => enemyAction.type === 'melee')
      .reduce((max, enemyAction) => Math.max(max, enemyAction.reach ?? 5), 5);
    return creatureDistance(enemy, active) <= reach;
  });
}

function pushInvalidActionLog(state: BattleState, active: Creature, agent: Agent, actionId: string): void {
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'Invalid Action',
    details: `${agent.id} requested ${actionId}, but the action could not be applied.`,
    type: 'info',
  });
}

function finishManualTurnEnd(state: BattleState, active: Creature): void {
  processHydraEndOfTurn(state, active);
  processTargetTurnEndOngoingEffects(state, active);
}

function applyLegalAction(
  state: BattleState,
  active: Creature,
  action: LegalAction,
  agent: Agent,
  turnStartAlreadyProcessed = false,
): void {
  if (action.type === 'battlecast_tactic') {
    applyBattlecastTacticTurn(state, active, action.tactic, turnStartAlreadyProcessed);
    return;
  }

  if (action.type === 'attack') {
    const target = state.creatures.find((creature) => creature.id === action.targetId);
    const battlecastAction = getActiveActions(active).find((candidate) => candidate.name === action.actionName);
    if (!target || !battlecastAction) {
      pushLog(state, {
        round: state.round,
        turn: state.turnIndex,
        actor: active.displayName,
        action: 'Invalid Action',
        details: `${agent.id} requested ${action.id}, but target or action was unavailable.`,
        type: 'info',
      });
      return;
    }
    resolveAttack(state, active, target, battlecastAction);
    active.hasActed = true;
    return;
  }

  if (action.type === 'move_toward') {
    const target = state.creatures.find((creature) => creature.id === action.targetId);
    if (!target) return;
    const from = { ...active.position };
    const destination = moveToward(active, target.position, state);
    active.position = destination;
    active.hasMovedThisTurn = creatureDistance({ ...active, position: from }, active) > 0;
    if (from.x !== destination.x || from.y !== destination.y) {
      pushLog(state, {
        round: state.round,
        turn: state.turnIndex,
        actor: active.displayName,
        action: 'Move',
        details: `${active.displayName} moves toward ${target.displayName} (${from.x},${from.y} -> ${destination.x},${destination.y}).`,
        type: 'move',
      });
    }
    return;
  }

  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: active.displayName,
    action: 'End Turn',
    details: `${active.displayName} ends their turn.`,
    type: 'info',
  });
}

function finishByRemainingHp(state: BattleState): void {
  const redHp = remainingHp(state, 'red');
  const blueHp = remainingHp(state, 'blue');
  state.isComplete = true;
  state.winner = redHp > blueHp ? 'red' : blueHp > redHp ? 'blue' : 'draw';
  pushLog(state, {
    round: state.round,
    turn: state.turnIndex,
    actor: 'System',
    action: 'Round Limit',
    details: `Round limit reached. ${state.winner === 'draw' ? 'Draw' : `${state.winner} wins`} by remaining HP (${redHp} red vs ${blueHp} blue).`,
    type: 'info',
  });
}

function remainingHp(state: BattleState, team: 'red' | 'blue'): number {
  return state.creatures
    .filter((creature) => creature.team === team && creature.isAlive)
    .reduce((sum, creature) => sum + Math.max(0, creature.currentHp), 0);
}

function scoreForWinner(winner: BattleState['winner'], team: 'red' | 'blue'): number {
  if (winner === 'draw' || winner === null) return 0.5;
  return winner === team ? 1 : 0;
}

function createMatchId(spec: AgentMatchSpec): string {
  return [
    spec.scenario.id,
    ...(spec.llmActionSpace && spec.llmActionSpace !== 'primitive' ? [spec.llmActionSpace] : []),
    String(spec.seed),
    spec.redAgent,
    spec.blueAgent,
  ].join('__').replace(/[^a-zA-Z0-9_.-]+/g, '_');
}
