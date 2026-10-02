import { TACTIC_LABELS, type BattleState, type TacticType } from './battlecast/engine/combat.js';
import { creatureDistance } from './battlecast/engine/combat.js';
import type { Creature } from './battlecast/types/monster.js';
import type { LegalAction, LegalActionCatalogue } from './legal-actions.js';
import type { RandomSource } from './random.js';
import { lk47Policies, type Lk47Policy } from './lk-47/variants.js';

export type BaselineAgentId =
  | 'baseline.random-legal'
  | 'baseline.nearest'
  | 'baseline.focus-fire'
  | 'baseline.expected-damage';
export type BattlecastTacticAgentId = `battlecast.${TacticType}`;
export type OpenRouterAgentId = `openrouter:${string}`;
export type Lk47AgentId = `lk-47${string}`;
export type AgentId = BaselineAgentId | BattlecastTacticAgentId | OpenRouterAgentId | Lk47AgentId;

export interface AgentDecisionContext {
  state: BattleState;
  activeCreature: Creature;
  catalogue: LegalActionCatalogue;
  rng: RandomSource;
}

export interface LegalActionAgent {
  kind: 'legal-action';
  id: AgentId;
  chooseAction(context: AgentDecisionContext): LegalAction;
}

export interface BattlecastTacticAgent {
  kind: 'battlecast-tactic';
  id: BattlecastTacticAgentId;
  tactic: TacticType;
  name: string;
  description: string;
}

export interface OpenRouterAgent {
  kind: 'openrouter-llm';
  id: OpenRouterAgentId;
  model: string;
}

export interface Lk47Agent {
  kind: 'lk-47';
  id: Lk47AgentId;
  policy: Lk47Policy;
}

export type Agent = LegalActionAgent | BattlecastTacticAgent | OpenRouterAgent | Lk47Agent;

export const baselineAgents: Record<BaselineAgentId, LegalActionAgent> = {
  'baseline.random-legal': {
    kind: 'legal-action',
    id: 'baseline.random-legal',
    chooseAction({ catalogue, rng }) {
      return catalogue.actions[Math.floor(rng.next() * catalogue.actions.length)] ?? { id: 'end_turn', type: 'end_turn' };
    },
  },
  'baseline.nearest': {
    kind: 'legal-action',
    id: 'baseline.nearest',
    chooseAction(context) {
      return pickNearestAction(context) ?? endTurn();
    },
  },
  'baseline.focus-fire': {
    kind: 'legal-action',
    id: 'baseline.focus-fire',
    chooseAction(context) {
      return pickFocusFireAction(context) ?? pickNearestAction(context) ?? endTurn();
    },
  },
  'baseline.expected-damage': {
    kind: 'legal-action',
    id: 'baseline.expected-damage',
    chooseAction(context) {
      return pickExpectedDamageAction(context) ?? pickFocusFireAction(context) ?? pickNearestAction(context) ?? endTurn();
    },
  },
};

export const battlecastTacticAgents: Record<BattlecastTacticAgentId, BattlecastTacticAgent> = {
  'battlecast.aggressive': createBattlecastTacticAgent('aggressive'),
  'battlecast.smart': createBattlecastTacticAgent('smart'),
  'battlecast.kiting': createBattlecastTacticAgent('kiting'),
  'battlecast.defensive': createBattlecastTacticAgent('defensive'),
};

export const agents: Record<AgentId, Agent> = {
  ...baselineAgents,
  ...battlecastTacticAgents,
};

export function getAgent(agentId: AgentId): Agent {
  if (isOpenRouterAgentId(agentId)) {
    return createOpenRouterAgent(agentId);
  }
  if (isLk47AgentId(agentId)) {
    return { kind: 'lk-47', id: agentId, policy: lk47Policies[agentId] };
  }
  const agent = agents[agentId];
  if (!agent) {
    throw new Error(`unknown agent: ${agentId}`);
  }
  return agent;
}

export function isAgentId(value: string): value is AgentId {
  return value in agents || isOpenRouterAgentId(value) || isLk47AgentId(value);
}

export function listAgentIds(): AgentId[] {
  return Object.keys(agents) as AgentId[];
}

export function listBattlecastTacticAgentIds(): BattlecastTacticAgentId[] {
  return Object.keys(battlecastTacticAgents) as BattlecastTacticAgentId[];
}

export function isOpenRouterAgentId(value: string): value is OpenRouterAgentId {
  return value.startsWith('openrouter:') && value.slice('openrouter:'.length).length > 0;
}

export function isLk47AgentId(value: string): value is Lk47AgentId {
  return Object.hasOwn(lk47Policies, value);
}

export function createOpenRouterAgentId(model: string): OpenRouterAgentId {
  return `openrouter:${model}` as OpenRouterAgentId;
}

function createBattlecastTacticAgent(tactic: TacticType): BattlecastTacticAgent {
  const label = TACTIC_LABELS[tactic];
  return {
    kind: 'battlecast-tactic',
    id: `battlecast.${tactic}` as BattlecastTacticAgentId,
    tactic,
    name: label.name,
    description: label.description,
  };
}

function createOpenRouterAgent(agentId: OpenRouterAgentId): OpenRouterAgent {
  return {
    kind: 'openrouter-llm',
    id: agentId,
    model: agentId.slice('openrouter:'.length),
  };
}

function pickNearestAction({ state, activeCreature, catalogue }: AgentDecisionContext): LegalAction | undefined {
  const enemiesByDistance = state.creatures
    .filter((creature) => creature.team !== activeCreature.team && creature.isAlive && !creature.dying)
    .sort((left, right) =>
      creatureDistance(activeCreature, left) - creatureDistance(activeCreature, right) ||
      left.displayName.localeCompare(right.displayName)
    );
  const nearest = enemiesByDistance[0];
  if (!nearest) return undefined;

  return catalogue.actions.find((action) => action.type === 'attack' && action.targetId === nearest.id)
    ?? catalogue.actions.find((action) => action.type === 'move_toward' && action.targetId === nearest.id);
}

function pickFocusFireAction({ state, catalogue }: AgentDecisionContext): LegalAction | undefined {
  const weakestReachable = state.creatures
    .filter((creature) => creature.isAlive && !creature.dying)
    .filter((creature) => catalogue.actions.some((action) =>
      action.type === 'attack' && action.targetId === creature.id
    ))
    .sort((left, right) =>
      left.currentHp - right.currentHp ||
      left.maxHp - right.maxHp ||
      left.displayName.localeCompare(right.displayName)
    )[0];

  if (!weakestReachable) return undefined;
  return catalogue.actions.find((action) => action.type === 'attack' && action.targetId === weakestReachable.id);
}

function pickExpectedDamageAction({ catalogue }: AgentDecisionContext): LegalAction | undefined {
  return catalogue.actions
    .filter((action): action is Extract<LegalAction, { type: 'attack' }> => action.type === 'attack')
    .sort((left, right) =>
      right.expectedDamage - left.expectedDamage ||
      left.targetName.localeCompare(right.targetName) ||
      left.actionName.localeCompare(right.actionName)
    )[0];
}

function endTurn(): LegalAction {
  return { id: 'end_turn', type: 'end_turn' };
}
